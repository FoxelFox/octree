# GPU Command Encoder & Shader Execution Flowchart

This document maps all GPUCommandEncoders and their associated shaders, organized by execution context and events.

## Overview

The application uses a deferred rendering pipeline with compute-based voxel generation, lighting, and culling. Command encoders are created at various points in the rendering loop and during async operations.

---

## 1. Main Rendering Loop (`index.ts`)

**Event:** `requestAnimationFrame` loop
**Location:** `src/index.ts:53-94`

### Command Encoder: `updateEncoder`
**Created:** `src/index.ts:67`

This encoder orchestrates the main per-frame updates for all active chunks.

**Execution Flow:**
```
loop() → streaming.update(updateEncoder) → device.queue.submit([updateEncoder.finish()])
```

---

## 2. Streaming System (`streaming.ts`)

### 2.1 Initial Chunk Generation

**Event:** Application initialization
**Location:** `src/chunk/streaming.ts:99-100`

#### Command Encoder: Noise Generation
**Created:** `src/chunk/streaming.ts:128`

**Pass Type:** Compute Pass
**Pipeline:** Noise generation
**Shader:** `noise.wgsl`
**Workgroups:** `ceil((gridSize+1)/4)³` (generates 257³ voxels)

**Execution:**
```
generateChunk() → noise.update(encoder, chunk)
```

#### Command Encoder: Mesh & Light Generation
**Created:** `src/chunk/streaming.ts:132`

**Pass 1 - Mesh Generation:**
- **Type:** Compute Pass
- **Pipeline:** Marching cubes mesh generation
- **Shader:** `mesh.wgsl`
- **Workgroups:** `ceil((gridSize/compression)/4)³`

**Pass 2 - Light Initialization:**
- **Type:** Compute Pass (16 iterations)
- **Pipeline:** Light propagation
- **Shader:** `light.wgsl`
- **Workgroups:** `ceil((gridSize/compression)/4)³`

**Execution:**
```
generateChunk() → mesh.update(meshEncoder, chunk)
                → light.update(meshEncoder, chunk)
```

### 2.2 Per-Frame Updates

**Event:** Main render loop
**Location:** `src/chunk/streaming.ts:202`

**Uses:** Main `updateEncoder` from `index.ts`

**Per-frame execution order:**

#### Pass 1: Light Propagation (per active chunk)
**Location:** `streaming.ts:255`
**Pipeline:** Light
**Shader:** `light.wgsl`
**Pass Type:** Compute Pass (16 iterations per chunk if invalidated)
**Workgroups:** `ceil((gridSize/compression)/4)³`

#### Pass 2: Frustum Culling (per active chunk)
**Location:** `streaming.ts:256`
**Pipeline:** Cull
**Shader:** `cull.wgsl`
**Pass Type:** Compute Pass
**Workgroups:** `ceil((gridSize/compression)/4)³`

#### Pass 3: Block Rendering (all active chunks)
**Location:** `streaming.ts:260`
**Pipeline:** Block (deferred rendering)
**Shaders:** `block.wgsl`, `block_deferred.wgsl`, `space_background.wgsl`

**Sub-passes:**
1. **Space Background Generation** (once on init)
   - Type: Compute Pass
   - Shader: `space_background.wgsl`
   - Workgroups: `256×128` (2048×1024 texture, 8×8 workgroup size)

2. **G-Buffer Pass** (all chunks together)
   - Type: Render Pass
   - Shader: `block.wgsl` (vertex + fragment)
   - Outputs: Position (rgba32float), Normal (rgba16float), Diffuse (rgba8unorm), Depth (depth24plus)

3. **Deferred Lighting Pass** (per chunk)
   - Type: Render Pass (multiple, one per chunk)
   - Shader: `block_deferred.wgsl` (vertex + fragment)
   - Output: Canvas texture (bgra8unorm)

---

## 3. Voxel Editor (`voxel_editor.ts`)

### 3.1 Position Reading

**Event:** User query for voxel position
**Location:** `src/pipeline/generation/voxel_editor.ts:87`

#### Command Encoder: Position Read
**Created:** `voxel_editor.ts:87`
**Label:** "Position Read"

**Operations:**
- Copy center pixel from G-buffer position texture to 1×1 texture
- Copy texture to CPU-readable buffer

**No shader execution** (texture/buffer copies only)

### 3.2 Voxel Editing

**Event:** User adds/removes voxels
**Location:** `src/pipeline/generation/voxel_editor.ts:379`

#### Command Encoder: Voxel Edit
**Created:** `voxel_editor.ts:379`
**Label:** "Voxel Edit"

**Pass Type:** Compute Pass
**Pipeline:** Voxel edit
**Shader:** `voxel_edit.wgsl`
**Workgroups:** `ceil((gridSize+1)/4)³` (operates on 257³ voxel grid)

**Timing:** Has timestamp writes

**Execution:**
```
User edit → queueEdit() → executeEditCommand(encoder)
```

### 3.3 Async Mesh Regeneration

**Event:** After voxel edits complete
**Location:** `src/pipeline/generation/voxel_editor.ts:460`

#### Command Encoder: Async Mesh Regeneration
**Created:** `voxel_editor.ts:460`
**Label:** "Async Mesh Regeneration"

**Pass Type:** Compute Pass
**Pipeline:** Mesh generation
**Shader:** `mesh.wgsl`
**Workgroups:** Partial or full chunk (depends on change bounds)

**Execution:**
```
Voxel edit → scheduleAsyncMeshUpdate() → regenerateMeshesAsync(encoder)
           → mesh.update(encoder, chunk, localBounds)
           → light.invalidate(chunk)
```

---

## 4. Async Culling Readback (`cull.ts`)

**Event:** Every 2 frames per chunk
**Location:** `src/pipeline/rendering/cull.ts:274`

#### Command Encoder: Culling Readback
**Created:** `cull.ts:274`
**Label:** (unnamed)

**Operations:**
- Copy counter buffer to readback buffer
- Copy indices buffer to readback buffer

**No shader execution** (buffer copies only)

**Execution:**
```
update() → startAsyncReadback() → performAsyncReadback(encoder)
```

---

## Event → Encoder Summary

| Event | Encoder | Shaders Executed | Location |
|-------|---------|------------------|----------|
| **App Init** | Noise Generation | `noise.wgsl` | `streaming.ts:128` |
| **App Init** | Mesh & Light | `mesh.wgsl`, `light.wgsl` | `streaming.ts:132` |
| **Every Frame** | Main Update | `light.wgsl`, `cull.wgsl`, `space_background.wgsl` (once), `block.wgsl`, `block_deferred.wgsl` | `index.ts:67` |
| **User Edit** | Voxel Edit | `voxel_edit.wgsl` | `voxel_editor.ts:379` |
| **Post-Edit** | Async Mesh Regen | `mesh.wgsl` | `voxel_editor.ts:460` |
| **Position Query** | Position Read | None (copy ops) | `voxel_editor.ts:87` |
| **Every 2 Frames** | Culling Readback | None (copy ops) | `cull.ts:274` |

---

## Shader → Pipeline Mapping

| Shader | Pipeline Type | Workgroup Size | Entry Point | Used In |
|--------|---------------|----------------|-------------|---------|
| `noise.wgsl` | Compute | 4×4×4 | `main` | Noise generation |
| `mesh.wgsl` | Compute | 4×4×4 | `main` | Mesh generation, marching cubes |
| `light.wgsl` | Compute | 4×4×4 | `main` | Light propagation (floodfill) |
| `cull.wgsl` | Compute | 4×4×4 | `main` | Frustum culling |
| `space_background.wgsl` | Compute | 8×8 | (default) | Space skybox generation |
| `block.wgsl` | Render | N/A | vertex + fragment | G-buffer generation |
| `block_deferred.wgsl` | Render | N/A | vertex + fragment | Deferred lighting |
| `voxel_edit.wgsl` | Compute | 4×4×4 | `main` | Voxel add/remove operations |

---

## Pipeline Dependencies

```
Initialization:
  noise.wgsl → mesh.wgsl → light.wgsl

Per Frame:
  [Previous frame state]
    ↓
  light.wgsl (if invalidated)
    ↓
  cull.wgsl
    ↓
  space_background.wgsl (once only)
    ↓
  block.wgsl (G-buffer)
    ↓
  block_deferred.wgsl (per chunk)

User Edit:
  voxel_edit.wgsl
    ↓
  mesh.wgsl
    ↓
  light.wgsl (invalidated)
    ↓
  [back to per-frame pipeline]
```

---

## Notes

- **Chunk-based execution:** Most pipelines operate per-chunk with neighbor chunk data
- **Async operations:** Voxel editing and culling use async GPU→CPU readback
- **Deferred rendering:** G-buffer pass renders all chunks, then per-chunk lighting passes
- **Timing instrumentation:** Block, Light, Mesh, Cull, and VoxelEditor pipelines have GPU timestamp queries
- **Double buffering:** Light propagation uses `chunk.light` and `chunk.nextLight` with buffer swapping
