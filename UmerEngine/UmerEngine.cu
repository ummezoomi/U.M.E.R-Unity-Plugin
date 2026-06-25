// UmerEngine.cu - Production Grade Volumetric Engine
// Features: O(1) Physics, Correct Hashing, Surface Extraction, Bandwidth Optimization, Material Physics, Dynamic Spawning

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <cstring>
#include <algorithm> // For std::min

// --- CONFIGURATION ---
// 1 Million Voxel Capacity (Real Scale)
#define HASH_SIZE (1 << 20) 
#define EMPTY_KEY 0xFFFFFFFFFFFFFFFF

// Material IDs
#define MAT_SAND 1
#define MAT_WATER 2
#define MAT_JELLY 3
#define MAT_STONE 4

// --- DATA STRUCTURES ---

// 1. The Heavy Physics Voxel (Internal GPU Only)
struct Voxel {
    unsigned char material_id; 
    float vx, vy, vz;
    float sub_x, sub_y, sub_z;
};

// 2. The Lightweight Render Voxel (Sent to Unity)
// Optimized for PCI-e transfer (Compact)
struct RenderVoxel {
    float x, y, z;       // Position
    uint32_t material_id; // Material
    uint32_t visible_mask; // Bitmask of exposed faces (Top, Bottom, Left...)
};

// 3. Spawn Command (For safe CPU->GPU Initialization)
struct SpawnCommand {
    int x, y, z;
    int material_id;
};

// --- DEVICE FUNCTIONS ---

__device__ unsigned long long pack_coord(int x, int y, int z) { 
    // Offset by 100k to handle negative coordinates
    return ((unsigned long long)(x+100000)&0xFFFFF) | 
           (((unsigned long long)(y+100000)&0xFFFFF)<<20) | 
           (((unsigned long long)(z+100000)&0xFFFFF)<<40); 
}

__device__ void unpack_coord(unsigned long long k, int* x, int* y, int* z) { 
    *x=(int)(k&0xFFFFF)-100000; 
    *y=(int)((k>>20)&0xFFFFF)-100000; 
    *z=(int)((k>>40)&0xFFFFF)-100000; 
}

__device__ uint32_t hash_func(unsigned long long k) {
    k ^= k >> 33; k *= 0xff51afd7ed558ccd; 
    k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53; 
    return (uint32_t)(k % HASH_SIZE);
}

// Reusable Insert Function (The Core Logic)
// FIXED: Correctly handles updates vs new inserts to prevent heap corruption
__device__ void insert_into_map(
    unsigned long long key, Voxel data, 
    unsigned long long* keys, int* vals, Voxel* heap, 
    int* heap_counter
) {
    uint32_t h = hash_func(key);
    for(int i=0; i<128; i++) { // Probe up to 128 slots
        int slot = (h + i) % HASH_SIZE;
        unsigned long long old = atomicCAS(&keys[slot], EMPTY_KEY, key);
        
        // CASE 1: SUCCESSFUL CLAIM (We are the first)
        if (old == EMPTY_KEY) {
            int idx = atomicAdd(heap_counter, 1);
            if (idx >= HASH_SIZE) return; // Safety check
            vals[slot] = idx;
            heap[idx] = data;
            return;
        }

        // CASE 2: ALREADY EXISTS (Collision/Update)
        // We reuse the existing heap index instead of allocating a new one.
        if (old == key) {
            int idx = vals[slot];
            heap[idx] = data; // Overwrite existing data
            return;
        }
    }
}

// Check Neighbor (For Physics & Surface Detection)
// Returns index if found, -1 if empty
__device__ int get_neighbor_idx(int x, int y, int z, unsigned long long* keys, int* vals) {
    unsigned long long k = pack_coord(x, y, z);
    uint32_t h = hash_func(k);
    for(int i=0; i<32; i++) {
        int slot = (h + i) % HASH_SIZE;
        if (keys[slot] == k) return vals[slot];
        if (keys[slot] == EMPTY_KEY) return -1;
    }
    return -1;
}

__device__ bool has_neighbor(int x, int y, int z, unsigned long long* keys) {
    unsigned long long k = pack_coord(x, y, z);
    uint32_t h = hash_func(k);
    for(int i=0; i<32; i++) {
        int slot = (h + i) % HASH_SIZE;
        if (keys[slot] == k) return true;
        if (keys[slot] == EMPTY_KEY) return false;
    }
    return false;
}

// --- KERNELS ---

// 1. Initialization Kernel (Processes Spawn Queue)
__global__ void spawn_kernel(
    SpawnCommand* commands, int count,
    unsigned long long* keys, int* vals, Voxel* heap, int* heap_counter
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    SpawnCommand cmd = commands[idx];
    Voxel v;
    v.material_id = cmd.material_id;
    v.vx = 0; v.vy = 0; v.vz = 0;
    v.sub_x = 0; v.sub_y = 0; v.sub_z = 0;

    unsigned long long key = pack_coord(cmd.x, cmd.y, cmd.z);
    insert_into_map(key, v, keys, vals, heap, heap_counter);
}

// 2. Physics Kernel
__global__ void physics_kernel(
    unsigned long long* src_keys, int* src_vals, Voxel* src_heap,
    unsigned long long* dst_keys, int* dst_vals, Voxel* dst_heap,
    int* dst_heap_counter,
    int max_voxels, float dt
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_voxels) return;

    unsigned long long key = src_keys[idx];
    if (key == EMPTY_KEY) return;
    
    int heap_idx = src_vals[idx];
    Voxel me = src_heap[heap_idx];
    
    int x, y, z;
    unpack_coord(key, &x, &y, &z);

    // --- PHYSICS LOGIC START ---
    
    // Apply Global Gravity
    me.vy -= 9.8f * dt;

    // Material Specific Behavior
    if (me.material_id == MAT_JELLY) {
        // Jelly: Try to stay close to neighbors (Spring force)
        // Check 6 directions
        int neighbors = 0;
        float pull_x=0, pull_y=0, pull_z=0;
        
        // Simplified Logic: If I have a neighbor, pull slightly towards them 
        // to maintain cohesion.
        if(has_neighbor(x+1, y, z, src_keys)) { pull_x += 1.0f; neighbors++; }
        if(has_neighbor(x-1, y, z, src_keys)) { pull_x -= 1.0f; neighbors++; }
        if(has_neighbor(x, y+1, z, src_keys)) { pull_y += 1.0f; neighbors++; }
        if(has_neighbor(x, y-1, z, src_keys)) { pull_y -= 1.0f; neighbors++; }
        
        if (neighbors > 0) {
            float strength = 5.0f; // Spring strength
            me.vx += pull_x * strength * dt;
            me.vy += pull_y * strength * dt;
            me.vz += pull_z * strength * dt;
        }
    }
    else if (me.material_id == MAT_WATER) {
        // Water: Spread out if piled up (Pressure)
        int up_idx = get_neighbor_idx(x, y+1, z, src_keys, src_vals);
        if (up_idx != -1) {
            // Someone is on top of me! Move sideways randomly.
            // Pseudo-random using position
            if ((x+y+z)%2 == 0) me.vx += 2.0f * dt;
            else me.vx -= 2.0f * dt;
            
            if ((x*z)%2 == 0) me.vz += 2.0f * dt;
            else me.vz -= 2.0f * dt;
        }
    }
    else if (me.material_id == MAT_SAND) {
        // Sand: High friction, simple gravity (already applied)
        me.vx *= 0.98f; // Strong air resistance/friction
        me.vz *= 0.98f;
    }

    // --- INTEGRATION ---
    me.sub_x += me.vx * dt;
    me.sub_y += me.vy * dt;
    me.sub_z += me.vz * dt;

    int step_x = (int)me.sub_x;
    int step_y = (int)me.sub_y;
    int step_z = (int)me.sub_z;
    me.sub_x -= step_x; me.sub_y -= step_y; me.sub_z -= step_z;

    int tx = x + step_x; 
    int ty = y + step_y; 
    int tz = z + step_z;

    // Floor Collision
    if (ty < -10) {
        ty = -10;
        if (me.material_id == MAT_JELLY) {
            me.vy *= -0.8f; // Bouncy
        } else {
            me.vy *= -0.1f; // Dull thud
            me.vx *= 0.8f;  // Ground friction
            me.vz *= 0.8f;
        }
    }
    // --- PHYSICS LOGIC END ---

    // --- INSERT INTO NEXT FRAME ---
    unsigned long long new_key = pack_coord(tx, ty, tz);
    insert_into_map(new_key, me, dst_keys, dst_vals, dst_heap, dst_heap_counter);
}

// 3. Surface Extraction Kernel (The Render Optimizer)
__global__ void surface_extraction_kernel(
    unsigned long long* keys, int* vals, Voxel* heap,
    RenderVoxel* render_buffer, int* render_counter,
    int max_voxels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_voxels) return;

    unsigned long long key = keys[idx];
    if (key == EMPTY_KEY) return;

    int x, y, z;
    unpack_coord(key, &x, &y, &z);

    // Check 6 neighbors to build Visibility Mask
    // Bit 0: +X, Bit 1: -X, Bit 2: +Y, etc.
    uint32_t mask = 0;
    if (!has_neighbor(x+1, y, z, keys)) mask |= 1;
    if (!has_neighbor(x-1, y, z, keys)) mask |= 2;
    if (!has_neighbor(x, y+1, z, keys)) mask |= 4;
    if (!has_neighbor(x, y-1, z, keys)) mask |= 8;
    if (!has_neighbor(x, y, z+1, keys)) mask |= 16;
    if (!has_neighbor(x, y, z-1, keys)) mask |= 32;

    // If fully occluded (mask == 0), do NOT render
    if (mask == 0) return;

    // Atomic Add to get a safe write index in the render buffer
    int write_idx = atomicAdd(render_counter, 1);
    
    int heap_idx = vals[idx];
    Voxel v = heap[heap_idx];

    // Write compact data for Unity
    RenderVoxel rv;
    rv.x = (float)x; 
    rv.y = (float)y; 
    rv.z = (float)z;
    rv.material_id = v.material_id;
    rv.visible_mask = mask;

    render_buffer[write_idx] = rv;
}

// --- MANAGER CLASS ---

class EngineManager {
public:
    // Double Buffers
    unsigned long long *d_keys_A, *d_keys_B;
    int *d_vals_A, *d_vals_B;
    Voxel *d_heap_A, *d_heap_B;
    int *d_heap_counter; // Atomic counter for active voxels

    // Spawn Queue (FIXED: Using Static Buffer)
    SpawnCommand *d_spawn_queue;     // GPU side
    SpawnCommand *h_spawn_buffer;    // CPU side (Static, reusable)
    int spawn_queue_capacity = 1000000; // 1 Million Command Buffer

    // Render Buffers
    RenderVoxel *d_render_buffer; // GPU side
    RenderVoxel *h_render_buffer; // CPU side (pinned)
    int *d_render_counter;        // Atomic counter for surface voxels
    int h_render_count_val = 0;   // How many to draw this frame

    int frame = 0;

    void Initialize() {
        // Allocations (1 Million Voxel Size)
        size_t map_bytes = HASH_SIZE * sizeof(unsigned long long);
        size_t val_bytes = HASH_SIZE * sizeof(int);
        size_t heap_bytes = HASH_SIZE * sizeof(Voxel);

        cudaMalloc(&d_keys_A, map_bytes); cudaMalloc(&d_keys_B, map_bytes);
        cudaMalloc(&d_vals_A, val_bytes); cudaMalloc(&d_vals_B, val_bytes);
        cudaMalloc(&d_heap_A, heap_bytes); cudaMalloc(&d_heap_B, heap_bytes);
        
        cudaMalloc(&d_heap_counter, sizeof(int));
        cudaMalloc(&d_render_counter, sizeof(int));
        
        cudaMalloc(&d_spawn_queue, spawn_queue_capacity * sizeof(SpawnCommand));
        
        // FIXED: Allocate Static CPU Buffer once
        cudaMallocHost(&h_spawn_buffer, spawn_queue_capacity * sizeof(SpawnCommand));

        cudaMalloc(&d_render_buffer, HASH_SIZE * sizeof(RenderVoxel));
        cudaMallocHost(&h_render_buffer, HASH_SIZE * sizeof(RenderVoxel));

        // Init Map A
        cudaMemset(d_keys_A, 0xFF, map_bytes);
        cudaMemset(d_heap_counter, 0, sizeof(int));
    }

    void ProcessSpawnQueue(SpawnCommand* host_cmds, int total_count) {
        // Process in chunks to respect spawn_queue_capacity
        int processed = 0;
        
        while (processed < total_count) {
            int current_batch_size = std::min(total_count - processed, spawn_queue_capacity);

            // Copy chunk to GPU
            cudaMemcpy(d_spawn_queue, &host_cmds[processed], current_batch_size * sizeof(SpawnCommand), cudaMemcpyHostToDevice);

            // Run Spawn Kernel
            int blocks = (current_batch_size + 255) / 256;
            spawn_kernel<<<blocks, 256>>>(
                d_spawn_queue, current_batch_size, 
                d_keys_A, d_vals_A, d_heap_A, d_heap_counter // Always spawn into A (assuming init or double buffer handled)
            );
            
            processed += current_batch_size;
        }
        
        cudaDeviceSynchronize();
    }

    void Step(float dt) {
        auto in_k = (frame % 2 == 0) ? d_keys_A : d_keys_B;
        auto in_v = (frame % 2 == 0) ? d_vals_A : d_vals_B;
        auto in_heap = (frame % 2 == 0) ? d_heap_A : d_heap_B;
        
        auto out_k = (frame % 2 == 0) ? d_keys_B : d_keys_A;
        auto out_v = (frame % 2 == 0) ? d_vals_B : d_vals_A;
        auto out_heap = (frame % 2 == 0) ? d_heap_B : d_heap_A;

        // 1. Reset Destination & Counters
        cudaMemset(out_k, 0xFF, HASH_SIZE * sizeof(unsigned long long));
        cudaMemset(d_heap_counter, 0, sizeof(int)); // Reset heap counter for next frame

        // 2. Physics Kernel
        int blocks = (HASH_SIZE + 255) / 256;
        physics_kernel<<<blocks, 256>>>(in_k, in_v, in_heap, out_k, out_v, out_heap, d_heap_counter, HASH_SIZE, dt);
        
        // 3. Surface Extraction (Render Prep)
        cudaMemset(d_render_counter, 0, sizeof(int)); // Reset render count
        surface_extraction_kernel<<<blocks, 256>>>(out_k, out_v, out_heap, d_render_buffer, d_render_counter, HASH_SIZE);

        // 4. Copy ONLY Surface Voxels to CPU
        // Read the atomic counter to know exactly how many voxels to copy
        cudaMemcpy(&h_render_count_val, d_render_counter, sizeof(int), cudaMemcpyDeviceToHost);
        
        if (h_render_count_val > 0) {
            cudaMemcpy(h_render_buffer, d_render_buffer, h_render_count_val * sizeof(RenderVoxel), cudaMemcpyDeviceToHost);
        }

        frame++;
    }

    void Cleanup() {
        cudaFree(d_keys_A); cudaFree(d_keys_B);
        cudaFree(d_vals_A); cudaFree(d_vals_B);
        cudaFree(d_heap_A); cudaFree(d_heap_B);
        cudaFree(d_heap_counter);
        cudaFree(d_render_counter);
        cudaFree(d_spawn_queue);
        cudaFree(d_render_buffer);
        
        // FIXED: Free static buffer
        cudaFreeHost(h_spawn_buffer);
        cudaFreeHost(h_render_buffer);
    }
};

EngineManager* g_Engine = nullptr;

// --- EXPORTS ---
extern "C" {
    __declspec(dllexport) void StartEngine() {
        if (!g_Engine) {
            g_Engine = new EngineManager();
            g_Engine->Initialize();
        }
    }

    // FIXED: Uses Static CPU Buffer + Flushing Logic (Safe for large volumes)
    __declspec(dllexport) void SpawnVolume(int x, int y, int z, int width, int height, int depth, int matId) {
        if (g_Engine) {
            int idx = 0;
            
            for(int ix=0; ix<width; ix++) {
                for(int iy=0; iy<height; iy++) {
                    for(int iz=0; iz<depth; iz++) {
                        
                        g_Engine->h_spawn_buffer[idx].x = x + ix;
                        g_Engine->h_spawn_buffer[idx].y = y + iy;
                        g_Engine->h_spawn_buffer[idx].z = z + iz;
                        g_Engine->h_spawn_buffer[idx].material_id = matId;
                        idx++;

                        // If buffer is full, flush to GPU and reset
                        if (idx >= g_Engine->spawn_queue_capacity) {
                            g_Engine->ProcessSpawnQueue(g_Engine->h_spawn_buffer, idx);
                            idx = 0;
                        }
                    }
                }
            }
            // Flush remaining commands
            if (idx > 0) {
                g_Engine->ProcessSpawnQueue(g_Engine->h_spawn_buffer, idx);
            }
        }
    }

    __declspec(dllexport) void UpdateEngine(float dt) {
        if (g_Engine) g_Engine->Step(dt);
    }

    // New Safe Accessor
    __declspec(dllexport) RenderVoxel* GetRenderData(int* count) {
        if (g_Engine) {
            *count = g_Engine->h_render_count_val;
            return g_Engine->h_render_buffer;
        }
        *count = 0;
        return nullptr
    }

    __declspec(dllexport) void StopEngine() { 
        if (g_Engine) {
            g_Engine->Cleanup();
            delete g_Engine;
            g_Engine = nullptr;
        }
    }
}