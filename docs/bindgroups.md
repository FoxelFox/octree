# WebGPU Buffer Relationships

This note reorganizes the `docs/bindgroups.puml` information around GPU resources so you can see who creates a buffer/texture and which pipeline later consumes it.

## Buffer & Texture Summary

| Resource | Type | Produced by | Consumed by | Notes |
| --- | --- | --- | --- | --- |
| `noiseBuffer` | GPUBuffer (storage) | `NoisePipeline` (writes noise), `VoxelEditPipeline` (mutates) | `MeshPipeline` | Central noise field shared between generation and editing.
| `vertexCounts` | GPUBuffer (storage out) | `MeshPipeline` | `CullPipeline` | Per-chunk vertex counts for indirect draw prep.
| `vertices` | GPUBuffer (storage → vertex) | `MeshPipeline` | `GBufferPipeline` | Promoted to vertex input during block G-buffer pass.
| `normals` | GPUBuffer (storage → vertex) | `MeshPipeline` | `GBufferPipeline` | Used as vertex attribute in G-buffer.
| `colors` | GPUBuffer (storage → vertex) | `MeshPipeline` | `GBufferPipeline` | Provides per-vertex color data.
| `density` | GPUBuffer (storage out) | `MeshPipeline` | `LightPipeline`, `CullPipeline` | Acts as voxel occupancy/lighting source data.
| `commands` | GPUBuffer (storage / indirect) | `MeshPipeline` | (Indirect draw encoder, not shown) | Packed indirect draw commands.
| `nextLightBuffer` | GPUBuffer (storage out) | `LightPipeline` | (Next lighting pass, not shown) | Holds frontier nodes for flood-fill.
| `lightBuffer` | GPUBuffer (storage lighting) | `LightPipeline` | `DeferredPipeline` | Double-buffered lighting results exposed via `light.getLightBuffer()`.
| `counter` | GPUBuffer (storage read/write) | `CullPipeline` | `CullPipeline` | Atomic counter used inside the cull pass.
| `indicesBuffer` | GPUBuffer (storage out) | `CullPipeline` | (Indexed draw, not shown) | Generated index list post-cull.
| `spaceBackgroundTexture` | StorageTexture | `SpaceBackgroundPipeline` | `DeferredPipeline` | Background skybox/light probe.
| `positionTexture` | TextureView (sampled) | `GBufferPipeline` | `DeferredPipeline` | G-buffer position target.
| `normalTexture` | TextureView (sampled) | `GBufferPipeline` | `DeferredPipeline` | G-buffer normal target.
| `diffuseTexture` | TextureView (sampled) | `GBufferPipeline` | `DeferredPipeline` | G-buffer albedo target.
| `depthTexture` | TextureView (sampled) | `GBufferPipeline` | `DeferredPipeline` | Captured depth for lighting/compositing.

## Resource Flow

```mermaid
flowchart LR
  classDef pipeline fill:#e0f2fe,stroke:#1d4ed8,color:#0f172a,stroke-width:1px
  classDef buffer fill:#fef3c7,stroke:#b45309,color:#78350f,stroke-width:1px
  classDef texture fill:#ede9fe,stroke:#6d28d9,color:#4c1d95,stroke-width:1px

  subgraph Pipelines
    NoisePipeline[["NoisePipeline\nCompute"]]
    VoxelEditPipeline[["VoxelEditPipeline\nCompute"]]
    MeshPipeline[["MeshPipeline\nCompute"]]
    LightPipeline[["LightPipeline\nCompute"]]
    CullPipeline[["CullPipeline\nCompute"]]
    SpaceBackgroundPipeline[["SpaceBackgroundPipeline\nCompute"]]
    GBufferPipeline[["GBufferPipeline\nRender"]]
    DeferredPipeline[["DeferredPipeline\nRender"]]
  end

  subgraph Buffers
    noiseBuffer{{"noiseBuffer\nGPUBuffer"}}
    vertexCounts{{"vertexCounts\nGPUBuffer"}}
    vertices{{"vertices\nGPUBuffer"}}
    normals{{"normals\nGPUBuffer"}}
    colors{{"colors\nGPUBuffer"}}
    density{{"density\nGPUBuffer"}}
    commands{{"commands\nGPUBuffer"}}
    nextLightBuffer{{"nextLightBuffer\nGPUBuffer"}}
    lightBuffer{{"lightBuffer\nGPUBuffer"}}
    counter{{"counter\nGPUBuffer"}}
    indicesBuffer{{"indicesBuffer\nGPUBuffer"}}
  end

  subgraph Textures
    spaceBackgroundTexture{{"spaceBackgroundTexture\nStorageTexture"}}
    positionTexture{{"positionTexture\nTextureView"}}
    normalTexture{{"normalTexture\nTextureView"}}
    diffuseTexture{{"diffuseTexture\nTextureView"}}
    depthTexture{{"depthTexture\nTextureView"}}
  end

  NoisePipeline -->|"writes"| noiseBuffer
  VoxelEditPipeline <-->|"mutates"| noiseBuffer
  noiseBuffer -->|"storage in"| MeshPipeline

  MeshPipeline -->|"writes"| vertexCounts
  vertexCounts -->|"storage in"| CullPipeline

  MeshPipeline -->|"writes"| vertices
  MeshPipeline -->|"writes"| normals
  MeshPipeline -->|"writes"| colors
  vertices -->|"vertex input"| GBufferPipeline
  normals -->|"vertex input"| GBufferPipeline
  colors -->|"vertex input"| GBufferPipeline

  MeshPipeline -->|"writes"| density
  density -->|"storage in"| LightPipeline
  density -->|"storage in"| CullPipeline

  MeshPipeline -->|"writes"| commands

  LightPipeline -->|"writes"| nextLightBuffer
  LightPipeline -->|"writes"| lightBuffer
  lightBuffer -->|"lighting data"| DeferredPipeline

  CullPipeline <-->|"atomic"| counter
  CullPipeline -->|"writes"| indicesBuffer

  SpaceBackgroundPipeline -->|"writes"| spaceBackgroundTexture
  spaceBackgroundTexture -->|"sampled"| DeferredPipeline

  GBufferPipeline -->|"renders"| positionTexture
  GBufferPipeline -->|"renders"| normalTexture
  GBufferPipeline -->|"renders"| diffuseTexture
  GBufferPipeline -->|"renders"| depthTexture
  positionTexture -->|"sampled"| DeferredPipeline
  normalTexture -->|"sampled"| DeferredPipeline
  diffuseTexture -->|"sampled"| DeferredPipeline
  depthTexture -->|"sampled"| DeferredPipeline
```

## Shared Uniform Inputs

- `contextUniform.uniformBuffer` feeds camera/global state into almost every pipeline.
- `offsetBuffer`, `edgeTableBuffer`, `triangleTableBuffer`, `editParamsBuffer`, and `configBuffer` are CPU-authored uniforms referenced where noted in the original bind groups but omitted from the flow diagram for clarity.

> [!tip]
> Use Obsidian’s Mermaid preview to pan/zoom the graph when the buffer list grows; it helps keep long resource names legible.
