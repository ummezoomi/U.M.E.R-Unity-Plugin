// UmerBridge_V2.cs
// Unity-side glue for UmerEngine V2.
//
// How this differs from V1:
//   V1: Engine writes to CPU pinned memory -> Unity re-uploads to GPU every frame (2x PCIe)
//   V2: Engine writes directly into a GraphicsBuffer that never leaves VRAM (0 PCIe copies)
//
// Setup in Unity:
//   1. Attach this component to any GameObject.
//   2. Assign a Material using the UmerVoxel.shader (included below as a comment).
//   3. Press Play. CUDA and Unity share the same VRAM buffer automatically.

using System;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;

public class UmerBridge_V2 : MonoBehaviour
{
    // ── Inspector ────────────────────────────────────────────────────────────
    [Header("Rendering")]
    public Mesh      voxelMesh;       // Assign a 1x1x1 cube mesh
    public Material  voxelMaterial;   // Assign UmerVoxel.mat (reads the GraphicsBuffer)
    public int       maxSurfaceVoxels = 524288; // Must match MAX_SURFACE in .cu

    [Header("Simulation")]
    public float deltaTime   = 0.016f; // Fixed physics timestep
    public int   spawnWidth  = 100;
    public int   spawnHeight = 100;
    public int   spawnDepth  = 100;
    public int   spawnMat    = 1;      // 1=Sand, 2=Water, 3=Jelly, 4=Stone

    // ── DLL Imports ──────────────────────────────────────────────────────────
    const string DLL = "UmerEngine";

    [DllImport(DLL)] static extern bool StartEngine();
    [DllImport(DLL)] static extern bool RegisterRenderBuffer(IntPtr dxBufferPtr, int byteSize);
    [DllImport(DLL)] static extern void SpawnVolume(int x, int y, int z, int w, int h, int d, int mat);
    [DllImport(DLL)] static extern void UpdateEngine(float dt);
    [DllImport(DLL)] static extern int  GetSurfaceVoxelCount();
    [DllImport(DLL)] static extern void StopEngine();

    // ── Runtime State ────────────────────────────────────────────────────────

    // The shared VRAM buffer. CUDA writes to it; Unity's shader reads from it.
    // Stride = 32 bytes (matches VoxelRenderData in .cu: 3 floats pos + 2 uint + 2 float pad)
    GraphicsBuffer renderBuffer;
    const int STRIDE = 32;

    // Indirect draw args buffer (avoids any CPU readback for draw calls)
    GraphicsBuffer argsBuffer;
    uint[]         args = new uint[5] { 0, 0, 0, 0, 0 };

    bool engineReady = false;

    void OnEnable()
    {
        if (!StartEngine())
        {
            Debug.LogError("[UMER V2] CUDA engine failed to start. Check that UmerEngine.dll is in Assets/Plugins/.");
            return;
        }

        // ── Allocate the shared VRAM buffer ──────────────────────────────────
        // Target.Structured = StructuredBuffer in HLSL, mapped as ID3D11Buffer in DX11.
        renderBuffer = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured,
            maxSurfaceVoxels,
            STRIDE
        );

        // Hand the native DX11 pointer to CUDA — this is the one-time handshake.
        IntPtr nativePtr  = renderBuffer.GetNativeBufferPtr();
        int    bufferBytes = maxSurfaceVoxels * STRIDE;

        if (!RegisterRenderBuffer(nativePtr, bufferBytes))
        {
            Debug.LogError("[UMER V2] cudaGraphicsD3D11RegisterResource failed. " +
                           "Ensure Unity is using the DirectX 11 graphics API (Project Settings > Player).");
            renderBuffer.Dispose();
            renderBuffer = null;
            return;
        }

        // Tell the material where to find the position data
        voxelMaterial.SetBuffer("_VoxelData", renderBuffer);

        // ── Indirect draw args ────────────────────────────────────────────────
        // args[0] = index count per instance (from mesh)
        // args[1] = instance count (updated each frame from GetSurfaceVoxelCount)
        // args[2] = start index location
        // args[3] = base vertex location
        // args[4] = start instance location
        argsBuffer = new GraphicsBuffer(
            GraphicsBuffer.Target.IndirectArguments,
            1, 5 * sizeof(uint)
        );
        args[0] = (uint)(voxelMesh != null ? voxelMesh.GetIndexCount(0) : 36);
        args[1] = 0;
        args[2] = 0;
        args[3] = 0;
        args[4] = 0;
        argsBuffer.SetData(args);

        // ── Spawn initial volume ──────────────────────────────────────────────
        SpawnVolume(0, 50, 0, spawnWidth, spawnHeight, spawnDepth, spawnMat);

        engineReady = true;
        Debug.Log("[UMER V2] Engine ready. Zero-copy VRAM buffer registered.");
    }

    void Update()
    {
        if (!engineReady) return;

        // Step CUDA physics + surface extraction
        // CUDA writes directly into renderBuffer's VRAM — no copy involved.
        UpdateEngine(deltaTime);

        // Read the surface count (4 bytes PCIe — unavoidable but negligible)
        int count = GetSurfaceVoxelCount();
        if (count <= 0) return;

        // Update instance count in the indirect args buffer (GPU-side, no readback)
        args[1] = (uint)Mathf.Min(count, maxSurfaceVoxels);
        argsBuffer.SetData(args);

        // Draw — Unity reads positions from renderBuffer, applies PBR lighting, shadows, bloom.
        // No sorting. No manual matrix arrays. GPU-driven.
        Graphics.DrawMeshInstancedIndirect(
            voxelMesh,
            0,
            voxelMaterial,
            new Bounds(Vector3.zero, Vector3.one * 2000f),
            argsBuffer
        );
    }

    void OnDisable()
    {
        engineReady = false;
        StopEngine();
        renderBuffer?.Dispose();
        argsBuffer?.Dispose();
        renderBuffer = null;
        argsBuffer   = null;
    }
}

/* ─── UmerVoxel.shader ────────────────────────────────────────────────────────
   Copy this into a new shader file: Assets/Shaders/UmerVoxel.shader
   This reads the VoxelRenderData StructuredBuffer and renders each voxel
   as a GPU-instanced cube with Unity's standard PBR lighting.

Shader "UMER/VoxelInstanced"
{
    Properties
    {
        _SandColor  ("Sand Color",  Color) = (0.93, 0.79, 0.54, 1)
        _WaterColor ("Water Color", Color) = (0.13, 0.59, 0.95, 0.7)
        _JellyColor ("Jelly Color", Color) = (0.72, 0.23, 0.82, 0.8)
        _StoneColor ("Stone Color", Color) = (0.45, 0.45, 0.45, 1)
        _Smoothness ("Smoothness",  Range(0,1)) = 0.4
    }

    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" }
        LOD 200

        CGPROGRAM
        #pragma surface surf Standard fullforwardshadows
        #pragma instancing_options procedural:setup
        #pragma target 4.5

        // Matches VoxelRenderData in UmerEngine_V2.cu (32 bytes)
        struct VoxelRenderData {
            float3   position;
            uint     material_id;
            uint     visible_face_mask;
            float    pad0;
            float    pad1;
        };

        #ifdef UNITY_PROCEDURAL_INSTANCING_ENABLED
        StructuredBuffer<VoxelRenderData> _VoxelData;
        #endif

        fixed4 _SandColor, _WaterColor, _JellyColor, _StoneColor;
        float  _Smoothness;

        void setup()
        {
            #ifdef UNITY_PROCEDURAL_INSTANCING_ENABLED
            VoxelRenderData v = _VoxelData[unity_InstanceID];
            unity_ObjectToWorld._11_21_31_41 = float4(1, 0, 0, 0);
            unity_ObjectToWorld._12_22_32_42 = float4(0, 1, 0, 0);
            unity_ObjectToWorld._13_23_33_43 = float4(0, 0, 1, 0);
            unity_ObjectToWorld._14_24_34_44 = float4(v.position.x, v.position.y, v.position.z, 1);
            unity_WorldToObject = unity_ObjectToWorld;
            unity_WorldToObject._14_24_34 *= -1;
            #endif
        }

        struct Input { float3 worldPos; };

        void surf(Input IN, inout SurfaceOutputStandard o)
        {
            #ifdef UNITY_PROCEDURAL_INSTANCING_ENABLED
            VoxelRenderData v = _VoxelData[unity_InstanceID];
            fixed4 col;
            if      (v.material_id == 1) col = _SandColor;
            else if (v.material_id == 2) col = _WaterColor;
            else if (v.material_id == 3) col = _JellyColor;
            else                         col = _StoneColor;
            o.Albedo     = col.rgb;
            o.Alpha      = col.a;
            o.Smoothness = _Smoothness;
            #endif
        }
        ENDCG
    }
    FallBack "Diffuse"
}
*/
