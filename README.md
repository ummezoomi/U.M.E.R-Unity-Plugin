# U.M.E.R-Unity-Plugin
> *"Unity provides the glass. U.M.E.R. provides the physics. We do not simulate; we compute deterministically on bare metal."*

This repository houses the Native CUDA Plugin (`UmerEngine.dll`) that bridges the high-level Unity Engine with the bare-metal **U.M.E.R. (Uniform Memory Encoded Representation)** spatial compute architecture. 

By utilizing low-level `[DllImport]` hooks and native C++ to C# interoperability, this plugin entirely bypasses Unity's CPU-bound physics (PhysX) and standard Compute Shaders. It injects deterministic, sort-free $O(1)$ spatial intelligence directly into the GPU's VRAM.

---

### 👁️ The Dream: A Synthetic Reality Compiler
The modern AI and Computer Vision industries are starved for mathematically ground-truth data. Traditional engines use floating-point approximations, non-deterministic execution orders, and pseudo-random collision resolutions. 

**The goal of this plugin is to turn Unity into a "Synthetic Reality Compiler."**

By slaving Unity's rendering pipeline to the U.M.E.R. CUDA backend, we achieve:
1. **Absolute Determinism:** Running a chaotic fluid or 500,000-agent crowd simulation will yield the *exact* same bit-level results on frame 10,000 every single time.
2. **Zero-Copy VRAM Execution:** Unity allocates the memory buffer; U.M.E.R. takes the raw hardware pointer. No CPU-to-GPU PCIe bandwidth bottlenecks. The CPU merely orchestrates the frame loop.
3. **Training-Free Agentic AI:** Bypassing standard Unity NavMesh and ML-Agents. We map arbitrary 3D models into U.M.E.R.'s "Mass Gravity" topology, solving kinematic navigation in pure $O(1)$ time without probabilistic neural networks.

---

### ⚙️ Plugin Architecture

The repository is structured as a standard Visual Studio C++/CUDA dynamic library build targeting x64 architectures.

* **`UmerEngine.cu`:** The core interface file. This contains the `extern "C"` export functions that expose the U.M.E.R. Prefix-Sum memory compaction and autonomous VRAM defragmentation kernels to C#.
* **`UmerEngine.dll`:** The compiled native payload. Injected directly into the `Assets/Plugins/x86_64` directory of the Unity project.
* **Execution Flow:**
  1. Unity C# initializes an array of struct data (e.g., positions, velocities, bounding volumes).
  2. Unity pins the memory and passes the hardware pointer to the `.dll`.
  3. U.M.E.R. seizes the thread, executes warp-synchronized spatial hashing, updates the data in place, and releases the thread.
  4. Unity immediately renders the updated VRAM buffer using `Graphics.DrawMeshInstancedIndirect`.

---

### ⚡ The Bottleneck vs. The U.M.E.R. Solution

| The Industry Standard (Unity/PhysX) | The U.M.E.R. Native Plugin |
| :--- | :--- |
| Relies on hierarchical bounding trees (BVH) forcing $O(\log N)$ branch divergence on the GPU. | Utilizes a flat, pointerless Delta-Histogram topology achieving strict $O(1)$ spatial lookups. |
| Sorts agents every frame using global Radix Sorts ($O(N \log N)$), destroying L1 cache coherency. | Employs view-independent amortized spatial hashing. Only agents crossing spatial boundaries trigger memory updates. |
| Degrades exponentially when data clusters (The Pigeonhole Paradox). | Hardware-level collision tolerance allows infinite spatial domains to absorb localized density without crashing. |

### 🛠️ Build Instructions
> *Requires NVIDIA CUDA Toolkit and MSVC.*

1. Open `UmerEngine.slnx` in Visual Studio.
2. Ensure the build target is set to **Release / x64**.
3. Build the solution. The pipeline will invoke `nvcc` to compile the `.cu` files and link them into `UmerEngine.dll`.
4. Copy the resulting `.dll` into your Unity project's `Assets/Plugins` folder.

### 🔬 Supported Use Cases
* **Granular Fluid Dynamics:** Real-time dam-break simulations with strict thermodynamic energy conservation (Symplectic Euler integration).
* **Massive Swarm Kinematics:** 500K+ complex agents executing 3D collision avoidance at 125+ FPS.
* **Neural Rendering Data Prep:** Bypassing sorting bottlenecks to generate pixel-perfect Gaussian Splatting ground truths.
