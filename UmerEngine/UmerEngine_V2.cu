// UmerEngine_V2.cu - Zero-Copy Hybrid Architecture
// CRITICAL UPGRADE: Eliminates the PCIe round-trip bottleneck.
// Data stays in VRAM. CUDA writes to a DirectX GraphicsBuffer that Unity reads directly.
//
// Architecture:
//   V1: CUDA -> [PCIe to CPU] -> [PCIe back to GPU] -> Unity renders
//   V2: CUDA -> DirectX GraphicsBuffer (VRAM) -> Unity renders  (zero copy)
//
// Rendering path: Unity PBR + shadows + bloom (kept).
// Sorting:        None. Position-only buffer. Unity's instanced shader handles draw order.
// Physics:        O(1) spatial hash, unchanged from V1.

#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>   // DirectX 11 <-> CUDA interop
#include <device_launch_parameters.h>
#include <stdint.h>
#include <cstring>
#include <algorithm>

// ─── CONFIGURATION ───────────────────────────────────────────────────────────
#define HASH_SIZE       (1 << 20)           // 1M voxel capacity
#define EMPTY_KEY       0xFFFFFFFFFFFFFFFF
#define MAX_SURFACE     524288              // Max visible voxels (2^19, tune per scene)

// Material IDs (keep parity with V1 + Unity MaterialPropertyBlock)
#define MAT_SAND   1
#define MAT_WATER  2
#define MAT_JELLY  3
#define MAT_STONE  4

// ─── DATA STRUCTURES ─────────────────────────────────────────────────────────

// Internal heavy voxel — stays in VRAM only, never crosses PCIe
struct Voxel {
    unsigned char material_id;
    float vx, vy, vz;          // velocity
    float sub_x, sub_y, sub_z; // sub-voxel displacement
};

// The render payload written into the shared DirectX/CUDA buffer.
// Unity's HLSL shader reads this via StructuredBuffer<VoxelRenderData>.
// Kept 32 bytes (GPU cache-line friendly).
struct VoxelRenderData {
    float    x, y, z;          // world position
    uint32_t material_id;       // maps to Unity MaterialPropertyBlock
    uint32_t visible_face_mask; // bitmask: which faces are exposed (for triplanar UV)
    float    pad0, pad1;        // explicit padding to 32 bytes
};

// Spawn command — short-lived, CPU->GPU once at spawn time only
struct SpawnCommand {
    int x, y, z;
    int material_id;
};

// ─── DEVICE HELPERS ──────────────────────────────────────────────────────────

__device__ __forceinline__
unsigned long long pack_coord(int x, int y, int z) {
    return ((unsigned long long)(x + 100000) & 0xFFFFF)
         | (((unsigned long long)(y + 100000) & 0xFFFFF) << 20)
         | (((unsigned long long)(z + 100000) & 0xFFFFF) << 40);
}

__device__ __forceinline__
void unpack_coord(unsigned long long k, int* x, int* y, int* z) {
    *x = (int)(k         & 0xFFFFF) - 100000;
    *y = (int)((k >> 20) & 0xFFFFF) - 100000;
    *z = (int)((k >> 40) & 0xFFFFF) - 100000;
}

__device__ __forceinline__
uint32_t hash_func(unsigned long long k) {
    k ^= k >> 33; k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53ULL;
    return (uint32_t)(k % HASH_SIZE);
}

__device__ void insert_into_map(
    unsigned long long key, Voxel data,
    unsigned long long* keys, int* vals, Voxel* heap, int* heap_counter
) {
    uint32_t h = hash_func(key);
    for (int i = 0; i < 128; i++) {
        int slot = (h + i) % HASH_SIZE;
        unsigned long long old = atomicCAS(&keys[slot], EMPTY_KEY, key);
        if (old == EMPTY_KEY) {
            int idx = atomicAdd(heap_counter, 1);
            if (idx >= HASH_SIZE) return;
            vals[slot] = idx;
            heap[idx]  = data;
            return;
        }
        if (old == key) {
            heap[vals[slot]] = data;
            return;
        }
    }
}

__device__ __forceinline__
bool has_neighbor(int x, int y, int z, unsigned long long* keys) {
    unsigned long long k = pack_coord(x, y, z);
    uint32_t h = hash_func(k);
    for (int i = 0; i < 32; i++) {
        int slot = (h + i) % HASH_SIZE;
        if (keys[slot] == k)        return true;
        if (keys[slot] == EMPTY_KEY) return false;
    }
    return false;
}

// ─── KERNELS ─────────────────────────────────────────────────────────────────

// Spawn kernel: initialise voxels from a batch of SpawnCommands
__global__ void spawn_kernel(
    SpawnCommand* commands, int count,
    unsigned long long* keys, int* vals, Voxel* heap, int* heap_counter
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    SpawnCommand cmd = commands[idx];
    Voxel v = {};
    v.material_id = (unsigned char)cmd.material_id;

    insert_into_map(pack_coord(cmd.x, cmd.y, cmd.z), v, keys, vals, heap, heap_counter);
}

// Physics kernel: double-buffered position/velocity integration
__global__ void physics_kernel(
    unsigned long long* src_keys, int* src_vals, Voxel* src_heap,
    unsigned long long* dst_keys, int* dst_vals, Voxel* dst_heap,
    int* dst_heap_counter, int max_voxels, float dt
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_voxels) return;

    unsigned long long key = src_keys[idx];
    if (key == EMPTY_KEY) return;

    int x, y, z;
    unpack_coord(key, &x, &y, &z);

    int heap_idx = src_vals[idx];
    if (heap_idx < 0 || heap_idx >= max_voxels) return;

    Voxel v = src_heap[heap_idx];

    // Apply gravity
    v.vy -= 9.81f * dt;

    // Sub-voxel displacement
    v.sub_x += v.vx * dt;
    v.sub_y += v.vy * dt;
    v.sub_z += v.vz * dt;

    // Accumulate integer movement from sub-voxel displacement
    int move_x = (int)v.sub_x;
    int move_y = (int)v.sub_y;
    int move_z = (int)v.sub_z;
    v.sub_x -= (float)move_x;
    v.sub_y -= (float)move_y;
    v.sub_z -= (float)move_z;

    int nx = x + move_x;
    int ny = y + move_y;
    int nz = z + move_z;

    // Collision: try to move; cancel velocity if blocked
    if (has_neighbor(nx, y, z, src_keys)) { nx = x; v.vx = 0.f; v.sub_x = 0.f; }
    if (has_neighbor(nx, ny, z, src_keys)) { ny = y; v.vy = 0.f; v.sub_y = 0.f; }
    if (has_neighbor(nx, ny, nz, src_keys)) { nz = z; v.vz = 0.f; v.sub_z = 0.f; }

    insert_into_map(pack_coord(nx, ny, nz), v, dst_keys, dst_vals, dst_heap, dst_heap_counter);
}

// ─── V2 SURFACE EXTRACTION ───────────────────────────────────────────────────
// KEY CHANGE FROM V1:
//   V1 wrote to h_render_buffer (pinned CPU memory) via cudaMemcpy.
//   V2 writes DIRECTLY into the shared DirectX GraphicsBuffer (d_interop_render_buffer),
//   which is already in VRAM. Zero PCIe transfer. Unity reads it where it sits.
__global__ void surface_extraction_kernel_v2(
    unsigned long long* keys, int* vals, Voxel* heap,
    VoxelRenderData* d_interop_render_buffer,  // <-- THIS is the DirectX GraphicsBuffer mapped into CUDA
    int* render_counter,
    int max_slots
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_slots) return;

    unsigned long long key = keys[idx];
    if (key == EMPTY_KEY) return;

    int x, y, z;
    unpack_coord(key, &x, &y, &z);

    // Build face visibility bitmask (6 faces)
    // Bit 0=+Y, 1=-Y, 2=+X, 3=-X, 4=+Z, 5=-Z
    uint32_t mask = 0;
    if (!has_neighbor(x, y + 1, z, keys)) mask |= (1 << 0);
    if (!has_neighbor(x, y - 1, z, keys)) mask |= (1 << 1);
    if (!has_neighbor(x + 1, y, z, keys)) mask |= (1 << 2);
    if (!has_neighbor(x - 1, y, z, keys)) mask |= (1 << 3);
    if (!has_neighbor(x, y, z + 1, keys)) mask |= (1 << 4);
    if (!has_neighbor(x, y, z - 1, keys)) mask |= (1 << 5);

    // Interior voxel — skip, do not write to buffer at all
    if (mask == 0) return;

    int write_idx = atomicAdd(render_counter, 1);
    if (write_idx >= MAX_SURFACE) return; // Guard against overflow

    int heap_idx = vals[idx];
    Voxel v = heap[heap_idx];

    VoxelRenderData rd;
    rd.x                = (float)x;
    rd.y                = (float)y;
    rd.z                = (float)z;
    rd.material_id      = (uint32_t)v.material_id;
    rd.visible_face_mask = mask;
    rd.pad0             = 0.f;
    rd.pad1             = 0.f;

    // Direct write into DirectX buffer — no cudaMemcpy, no PCIe
    d_interop_render_buffer[write_idx] = rd;
}

// ─── ENGINE MANAGER ──────────────────────────────────────────────────────────

class EngineManager {
public:
    // Double-buffered spatial hash
    unsigned long long *d_keys_A, *d_keys_B;
    int                *d_vals_A, *d_vals_B;
    Voxel              *d_heap_A, *d_heap_B;
    int                *d_heap_counter;

    // Spawn queue (CPU -> GPU, only at spawn time)
    SpawnCommand *d_spawn_queue;
    SpawnCommand *h_spawn_buffer;
    int           spawn_queue_capacity = 1000000;

    // ── INTEROP STATE ──
    // The DirectX buffer is allocated by Unity (C#: new GraphicsBuffer(...))
    // Unity passes its native pointer to RegisterDirectXBuffer().
    // CUDA maps it once; we keep the mapped CUDA pointer for the lifetime of the session.
    cudaGraphicsResource_t dx_render_resource = nullptr; // DX11 <-> CUDA handle
    VoxelRenderData*       d_interop_render_buffer = nullptr; // CUDA-side ptr into DX buffer
    int*                   d_render_counter = nullptr;
    int                    h_render_count   = 0;

    int frame = 0;

    bool Initialize() {
        size_t map_bytes  = HASH_SIZE * sizeof(unsigned long long);
        size_t val_bytes  = HASH_SIZE * sizeof(int);
        size_t heap_bytes = HASH_SIZE * sizeof(Voxel);

        if (cudaMalloc(&d_keys_A, map_bytes)  != cudaSuccess) return false;
        if (cudaMalloc(&d_keys_B, map_bytes)  != cudaSuccess) return false;
        if (cudaMalloc(&d_vals_A, val_bytes)  != cudaSuccess) return false;
        if (cudaMalloc(&d_vals_B, val_bytes)  != cudaSuccess) return false;
        if (cudaMalloc(&d_heap_A, heap_bytes) != cudaSuccess) return false;
        if (cudaMalloc(&d_heap_B, heap_bytes) != cudaSuccess) return false;

        if (cudaMalloc(&d_heap_counter,   sizeof(int)) != cudaSuccess) return false;
        if (cudaMalloc(&d_render_counter, sizeof(int)) != cudaSuccess) return false;

        if (cudaMalloc(&d_spawn_queue, spawn_queue_capacity * sizeof(SpawnCommand)) != cudaSuccess) return false;
        if (cudaMallocHost(&h_spawn_buffer, spawn_queue_capacity * sizeof(SpawnCommand)) != cudaSuccess) return false;

        cudaMemset(d_keys_A,       0xFF, map_bytes);
        cudaMemset(d_keys_B,       0xFF, map_bytes);
        cudaMemset(d_heap_counter, 0,    sizeof(int));

        return true;
    }

    // ── Called once from C# after GraphicsBuffer is created ──
    // bufferPtr = (void*)buffer.GetNativeBufferPtr()  (from Unity C#)
    // bufferByteSize = buffer.count * buffer.stride   (must be >= MAX_SURFACE * sizeof(VoxelRenderData))
    bool RegisterDirectXBuffer(void* bufferPtr, size_t bufferByteSize) {
        if (dx_render_resource) {
            // Unregister previous if called again (e.g. buffer was resized)
            cudaGraphicsUnregisterResource(dx_render_resource);
            dx_render_resource = nullptr;
            d_interop_render_buffer = nullptr;
        }

        cudaError_t err = cudaGraphicsD3D11RegisterResource(
            &dx_render_resource,
            (ID3D11Buffer*)bufferPtr,           // The raw DX11 buffer pointer Unity gave us
            cudaGraphicsRegisterFlagsNone       // CUDA reads & writes allowed
        );
        return (err == cudaSuccess);
    }

    void ProcessSpawnQueue(SpawnCommand* cmds, int count) {
        int processed = 0;
        while (processed < count) {
            int batch = std::min(count - processed, spawn_queue_capacity);
            cudaMemcpy(d_spawn_queue, &cmds[processed], batch * sizeof(SpawnCommand), cudaMemcpyHostToDevice);

            int blocks = (batch + 255) / 256;
            spawn_kernel<<<blocks, 256>>>(
                d_spawn_queue, batch,
                d_keys_A, d_vals_A, d_heap_A, d_heap_counter
            );
            processed += batch;
        }
        cudaDeviceSynchronize();
    }

    void Step(float dt) {
        if (!dx_render_resource) return; // Buffer not registered yet — skip frame

        auto in_k    = (frame % 2 == 0) ? d_keys_A : d_keys_B;
        auto in_v    = (frame % 2 == 0) ? d_vals_A : d_vals_B;
        auto in_heap = (frame % 2 == 0) ? d_heap_A : d_heap_B;
        auto out_k   = (frame % 2 == 0) ? d_keys_B : d_keys_A;
        auto out_v   = (frame % 2 == 0) ? d_vals_B : d_vals_A;
        auto out_heap= (frame % 2 == 0) ? d_heap_B : d_heap_A;

        cudaMemset(out_k,          0xFF, HASH_SIZE * sizeof(unsigned long long));
        cudaMemset(d_heap_counter, 0,    sizeof(int));
        cudaMemset(d_render_counter, 0,  sizeof(int));

        int blocks = (HASH_SIZE + 255) / 256;

        // 1. Physics (double-buffered)
        physics_kernel<<<blocks, 256>>>(
            in_k, in_v, in_heap,
            out_k, out_v, out_heap,
            d_heap_counter, HASH_SIZE, dt
        );

        // 2. Map the DirectX buffer into CUDA address space
        //    This is a lightweight VRAM-to-VRAM mapping — NOT a copy.
        cudaGraphicsMapResources(1, &dx_render_resource, 0);

        size_t mapped_size = 0;
        cudaGraphicsResourceGetMappedPointer(
            (void**)&d_interop_render_buffer,
            &mapped_size,
            dx_render_resource
        );

        // 3. Surface extraction writes DIRECTLY into DirectX buffer
        surface_extraction_kernel_v2<<<blocks, 256>>>(
            out_k, out_v, out_heap,
            d_interop_render_buffer,  // <-- zero-copy target
            d_render_counter,
            HASH_SIZE
        );

        // 4. Unmap — hand control back to DirectX / Unity
        cudaGraphicsUnmapResources(1, &dx_render_resource, 0);

        // 5. Read render count (needed by Unity to call DrawMeshInstancedIndirect)
        //    This IS a small PCIe transfer (4 bytes) — unavoidable, but negligible.
        cudaMemcpy(&h_render_count, d_render_counter, sizeof(int), cudaMemcpyDeviceToHost);

        frame++;
    }

    void Cleanup() {
        if (dx_render_resource) {
            cudaGraphicsUnregisterResource(dx_render_resource);
            dx_render_resource = nullptr;
        }
        cudaFree(d_keys_A);   cudaFree(d_keys_B);
        cudaFree(d_vals_A);   cudaFree(d_vals_B);
        cudaFree(d_heap_A);   cudaFree(d_heap_B);
        cudaFree(d_heap_counter);
        cudaFree(d_render_counter);
        cudaFree(d_spawn_queue);
        cudaFreeHost(h_spawn_buffer);
    }
};

// ─── SINGLETON ───────────────────────────────────────────────────────────────
static EngineManager* g_Engine = nullptr;

// ─── EXPORTS (C# P/Invoke) ───────────────────────────────────────────────────
extern "C" {

    __declspec(dllexport) bool StartEngine() {
        if (!g_Engine) {
            g_Engine = new EngineManager();
            if (!g_Engine->Initialize()) {
                delete g_Engine;
                g_Engine = nullptr;
                return false;
            }
        }
        return true;
    }

    // Call this once from C# after creating the GraphicsBuffer:
    //   GraphicsBuffer buf = new GraphicsBuffer(GraphicsBuffer.Target.Structured, MAX_SURFACE, stride);
    //   RegisterRenderBuffer(buf.GetNativeBufferPtr(), MAX_SURFACE * stride);
    __declspec(dllexport) bool RegisterRenderBuffer(void* dxBufferPtr, int byteSize) {
        if (!g_Engine) return false;
        return g_Engine->RegisterDirectXBuffer(dxBufferPtr, (size_t)byteSize);
    }

    __declspec(dllexport) void SpawnVolume(int x, int y, int z, int width, int height, int depth, int matId) {
        if (!g_Engine) return;
        int idx = 0;
        for (int ix = 0; ix < width;  ix++)
        for (int iy = 0; iy < height; iy++)
        for (int iz = 0; iz < depth;  iz++) {
            g_Engine->h_spawn_buffer[idx++] = { x+ix, y+iy, z+iz, matId };
            if (idx >= g_Engine->spawn_queue_capacity) {
                g_Engine->ProcessSpawnQueue(g_Engine->h_spawn_buffer, idx);
                idx = 0;
            }
        }
        if (idx > 0) g_Engine->ProcessSpawnQueue(g_Engine->h_spawn_buffer, idx);
    }

    __declspec(dllexport) void UpdateEngine(float dt) {
        if (g_Engine) g_Engine->Step(dt);
    }

    // Returns the number of surface voxels written this frame.
    // Unity uses this as the instanceCount in DrawMeshInstancedIndirect.
    // V2: No pointer returned — data lives in the DirectX buffer Unity already holds.
    __declspec(dllexport) int GetSurfaceVoxelCount() {
        return g_Engine ? g_Engine->h_render_count : 0;
    }

    __declspec(dllexport) void StopEngine() {
        if (g_Engine) {
            g_Engine->Cleanup();
            delete g_Engine;
            g_Engine = nullptr;
        }
    }
}
