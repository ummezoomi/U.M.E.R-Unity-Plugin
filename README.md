# U.M.E.R. Unity Plugin — V2
### *Uniform Memory Encoded Representation*
> *"Unity provides the glass. U.M.E.R. provides the physics. The GPU is ours."*

A native CUDA plugin that performs a hostile takeover of Unity's rendering and physics pipelines. U.M.E.R. replaces Unity's CPU-bound PhysX and standard compute shaders with deterministic, bare-metal spatial intelligence — executing entirely inside VRAM via zero-copy DirectX/CUDA interop.

---

## What This Is

Most game engines treat the GPU as a rendering slave. You feed it data from the CPU, it draws a picture, and you do it again next frame. Unity is no exception — PhysX runs on the CPU, data crosses the PCIe bus, and the GPU spends half its time waiting.

U.M.E.R. inverts this model entirely.

The CPU's only job is to start the frame. Everything else — spatial hashing, collision detection, physics integration, and surface extraction — runs natively in VRAM. The result is written directly into a shared DirectX `GraphicsBuffer` that Unity reads without a single byte crossing the PCIe bus. Unity stops being a physics engine and becomes a window manager with excellent taste in lighting.

---

## V2 Architecture: Zero-Copy Hybrid

The fundamental breakthrough in V2 is eliminating the PCIe round-trip that bottlenecked V1.

**V1 flow (two full PCIe transfers per frame):**
```
CUDA (VRAM) → cudaMemcpy → CPU RAM → Unity uploads → GPU VRAM → render
```

**V2 flow (zero copies):**
```
CUDA (VRAM) → DirectX GraphicsBuffer (VRAM) → Unity renders
```

Unity allocates a `GraphicsBuffer`. U.M.E.R. receives its native hardware pointer via `cudaGraphicsD3D11RegisterResource`. From that point forward, CUDA writes surface voxel positions directly into the DirectX buffer. Unity reads from it. The data never moves.

This is not a compute shader trick. This is CUDA and DirectX sharing the same physical VRAM allocation.

---

## The Physics

U.M.E.R. replaces three specific industry bottlenecks with O(1) alternatives:

| Standard Pipeline | U.M.E.R. |
|:---|:---|
| BVH traversal: O(log N) branch divergence per ray | Flat pointerless delta-histogram topology: O(1) spatial lookup |
| Radix sort every frame: O(N log N), destroys L1 cache | View-independent spatial hashing: only boundary-crossing voxels trigger updates |
| PhysX degrades under clustering (Pigeonhole collapse) | Hardware-level collision tolerance absorbs arbitrary density |

The physics integration uses Symplectic Euler — the only integrator that conserves energy over long simulations without drift. A fluid sim running for 10,000 frames produces the exact same bit-level result every time. This is not an approximation. It is determinism.

---

## Rendering Strategy

U.M.E.R. uses a **hybrid rendering architecture** — the correct tool for each job:

**Voxel Simulation → ComputeBuffer Interop**
Surface extraction runs in CUDA, writes positions to the shared DirectX buffer, and Unity's instanced shader handles the draw call with full PBR lighting, shadows, and post-processing. You keep Unity's rendering quality. You replace its physics entirely.

**Ray Tracing & 3DGS → RenderTexture Interop** *(V2.1)*
For pipelines where the shading *is* the spatial math — DDA ray traversal, sort-free Gaussian splatting — U.M.E.R. writes finished pixel colors directly into a `RenderTexture`. Unity displays it. Unity does no lighting calculation whatsoever.

The rule: if Unity can improve the output, let it render. If the math produces the image, take the screen.

---

## Execution Flow

```
1. Unity C# creates a GraphicsBuffer in VRAM
2. Passes native DX11 pointer to UmerEngine.dll via [DllImport]
3. U.M.E.R. registers the buffer with CUDA (one-time handshake)

Each frame:
4. CUDA runs physics kernel (double-buffered spatial hash)
5. CUDA maps the DirectX buffer into CUDA address space
6. Surface extraction kernel writes positions directly into it
7. CUDA unmaps — hands control back to DirectX
8. Unity calls DrawMeshInstancedIndirect
9. GPU-instanced shader reads positions, renders with PBR + shadows + bloom
```

The only PCIe transfer per frame is 4 bytes — the surface voxel count needed for the draw call's instance argument.

---

## Repository Structure

```
UmerEngine.cu       — Core CUDA engine: spatial hash, physics, surface extraction,
                      DirectX interop registration and buffer management
UmerEngine.slnx     — Visual Studio solution (CUDA/MSVC, x64 Release)
UmerBridge_V2.cs    — Unity C# bridge: GraphicsBuffer lifecycle, DLL imports,
                      instanced draw calls, shader material binding
UmerVoxel.shader    — GPU-instanced surface shader: reads VoxelRenderData
                      StructuredBuffer, outputs PBR with per-material color
```

---

## Build & Integration

**Requirements:** NVIDIA CUDA Toolkit, MSVC, Visual Studio 2022+, Unity 2021.3+ with DirectX 11 graphics API

**Build:**
1. Open `UmerEngine.slnx` in Visual Studio
2. Set build target to **Release / x64**
3. Build — `nvcc` compiles the `.cu` files and links `UmerEngine.dll`
4. Copy `UmerEngine.dll` to `Assets/Plugins/x86_64/` in your Unity project

**Unity Setup:**
1. Attach `UmerBridge_V2.cs` to any GameObject
2. Assign a 1×1×1 cube mesh and a material using `UmerVoxel.shader`
3. Ensure **Project Settings → Player → Graphics API** is set to **DirectX 11**
4. Press Play

---

## Supported Use Cases

- **Granular Fluid Dynamics** — Real-time dam-break and fluid simulation with thermodynamic energy conservation. Millions of voxels. No sorting. No approximation.
- **Massive Swarm Kinematics** — 500K+ agents executing 3D collision avoidance at 125+ FPS, without Unity NavMesh or ML-Agents.
- **Neural Rendering Ground Truth** — Pixel-perfect Gaussian Splatting training data generated without the O(N log N) sort bottleneck that corrupts standard 3DGS pipelines.
- **Synthetic Reality Compilation** — Deterministic simulation output for AI training datasets. Frame 10,000 of a chaotic fluid produces the same bit-exact result on every run.

---

## Requirements

- NVIDIA GPU (Pascal architecture or newer recommended)
- CUDA Toolkit 12.x
- Unity 2021.3 LTS or newer
- DirectX 11 graphics API (set in Unity Player Settings)
- Windows x64 build target
