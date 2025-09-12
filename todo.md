# Automatic Chunk System Implementation

## Overview
Restructure the current single 256³ world into an infinite world system with 256×256×256 voxel chunks. Each chunk contains 32³ meshlets (8×8×8 voxels each) with individual density and light cache buffers. Target ~18 active chunks with automatic loading/unloading.

## Current Architecture Analysis
- **Current**: Single 256³ world (gridSize=256, compression=8)
- **Current**: 32³ meshlets per world, each 8³ voxels 
- **Current**: Single density buffer (32³ floats), single light buffer (32³ × RG)
- **Target**: Multiple 256³ chunks, each with own buffers

## Core Architecture Changes

### 1. Chunk Management System
- [ ] **Create ChunkManager class** (`src/chunk/ChunkManager.ts`)
  - Track active chunks in `Map<string, Chunk>` (key: `"${x},${y},${z}"`)
  - Handle chunk coordinate ↔ world space conversion
  - Implement chunk visibility testing using camera frustum
  - Manage chunk lifecycle (UNLOADED → GENERATING → READY → UNLOADING)
  - LRU cache management for memory limits

- [ ] **Restructure Chunk class** (`src/chunk/chunk.ts`)
  - **Current**: Basic GPU buffers (meshes, commands, density, indices)  
  - **Add**: Chunk world coordinates (chunkX, chunkY, chunkZ)
  - **Add**: Chunk size constant = 256 voxels (maintains current world size)
  - **Add**: Internal meshlet count = 32³ (maintains current compression)
  - **Add**: Loading state enum and timestamp
  - **Add**: Per-chunk density buffer (32³ floats)
  - **Add**: Per-chunk light cache buffer (32³ × RG32Float)
  - **Add**: GPU buffer cleanup and destroy methods

### 2. Dynamic World Generation
- [ ] **Extend Noise class** (`src/pipeline/noise.ts`)
  - **Current**: Generates single 256³ noise buffer
  - **Add**: `generateChunkNoise(chunkX, chunkY, chunkZ)` method
  - **Add**: Chunk offset uniforms for world-space noise sampling
  - **Add**: Per-chunk noise buffer management
  - **Modify**: Make noise generation coordinate-aware (world coords vs local chunk coords)

- [ ] **Update Mesh pipeline** (`src/pipeline/mesh.ts`)
  - **Current**: Single mesh buffer for 32³ meshlets
  - **Add**: Per-chunk mesh generation with world offsets
  - **Add**: Chunk-specific bindGroups and uniform buffers
  - **Modify**: `update()` method to accept chunk parameter
  - **Add**: Batch processing of multiple chunks per frame (max 2-3)

### 3. Light System Restructure
- [ ] **Extend Light class** (`src/pipeline/light.ts`)
  - **Current**: Single light buffer (32³ × RG) with flood-fill propagation
  - **Add**: Per-chunk light cache (32³ × RG32Float each)
  - **Add**: Cross-chunk light propagation at chunk boundaries
  - **Add**: Chunk light invalidation and selective updates
  - **Add**: Light boundary handling (skylight propagation between chunks)
  - **Modify**: Initialize skylight per-chunk based on world height

### 4. Chunk Loading Strategy  
- [ ] **Implement distance-based loading**
  - Calculate distance from player chunk to candidate chunks
  - Load radius: 2-3 chunks (adjustable based on performance)
  - Unload radius: 4-5 chunks (with hysteresis to prevent thrashing)
  - Priority system: visible chunks > adjacent > distant

- [ ] **Add visibility culling**
  - Test chunk bounding boxes (256³ each) against camera frustum
  - Load visible chunks + 1 chunk ahead in movement/view direction
  - Skip generation for occluded chunks (behind player)

### 5. Memory Management
- [ ] **Implement chunk capacity limits**
  - Maximum active chunks: ~18 (configurable)
  - Per-chunk memory: ~64MB (meshes + density + light + noise)
  - Total memory budget: ~1.2GB for chunks
  - LRU eviction when exceeding limits

- [ ] **Add streaming queue system**
  - Async chunk generation queue (max 2-3 concurrent)
  - Frame-spread work distribution (avoid hitches)
  - Priority-based processing (visible chunks first)
  - Graceful degradation under load

### 6. Player Movement Integration
- [ ] **Update player tracking** (`src/data/context.ts`)
  - Track current player chunk coordinates
  - Detect chunk boundary crossings (256-voxel boundaries)
  - Trigger chunk load/unload events on movement
  - Calculate view direction for predictive loading

- [ ] **Enhance camera system** (`src/gpu.ts`)
  - Add `getPlayerChunk()` method 
  - Calculate frustum planes for chunk culling
  - Track movement velocity for predictive loading distance

### 7. Multi-Chunk Rendering
- [ ] **Update Cull system** (`src/pipeline/cull.ts`)
  - **Current**: Single cull pass for 32³ meshlets
  - **Add**: Multi-chunk culling with index merging
  - **Add**: Per-chunk cull results combination
  - **Add**: Chunk-relative indexing for mesh commands
  - **Modify**: Handle dynamic chunk count in render loop

- [ ] **Update Block system** (`src/pipeline/block.ts`)
  - **Current**: Renders single mesh buffer
  - **Add**: Multi-chunk G-buffer generation
  - **Add**: Chunk-aware mesh binding and drawing
  - **Modify**: Handle multiple light buffers in deferred pass
  - **Add**: Chunk boundary seamless rendering

### 8. Voxel Editor Integration
- [ ] **Update VoxelEditor** (`src/pipeline/voxel_editor.ts`)
  - **Add**: World position → chunk coordinate conversion
  - **Add**: Cross-chunk editing support (edit radius may span chunks)
  - **Modify**: Target specific chunk buffers for edits
  - **Add**: Multi-chunk mesh regeneration after edits
  - **Add**: Cross-chunk light propagation after voxel changes

### 9. Configuration Constants
- [ ] **Add chunk system settings** (`src/index.ts`)
  ```typescript
  export const CHUNK_SIZE = 256; // voxels per chunk dimension  
  export const CHUNK_MESHLETS = 32; // meshlets per chunk dimension (32³ total)
  export const MESHLET_SIZE = 8; // voxels per meshlet (8³)
  export const MAX_ACTIVE_CHUNKS = 18;
  export const CHUNK_LOAD_RADIUS = 2; // chunks
  export const CHUNK_UNLOAD_RADIUS = 4; // chunks  
  export const MAX_CONCURRENT_CHUNK_GENS = 2;
  ```

### 10. WGSL Shader Updates
- [ ] **Update compute shaders for chunk awareness**
  - **`noise.wgsl`**: Add chunk world offset uniforms
  - **`mesh.wgsl`**: Add chunk coordinate system handling
  - **`light.wgsl`**: Add cross-chunk boundary light propagation
  - **`cull.wgsl`**: Handle per-chunk meshlet indexing
  - **`block.wgsl`**: Support multi-chunk mesh rendering

### 11. Performance Optimizations
- [ ] **Frame-spread chunk processing**
  - Use `requestIdleCallback` for non-critical chunk ops
  - Budget GPU time per frame (e.g., max 2ms for chunk work)
  - Async/await patterns for chunk loading pipeline
  - Batch similar operations across chunks

- [ ] **Chunk state persistence**
  - Cache recently unloaded chunk data (compressed)
  - Fast reload for chunks returning to view
  - Lazy cleanup of GPU resources (frame delay)

## Implementation Notes

### Chunk Coordinate System
- **Chunk size**: 256³ voxels (same as current world)  
- **Chunk coords**: Integer (chunkX, chunkY, chunkZ)
- **World position**: `(chunkX * 256, chunkY * 256, chunkZ * 256)`
- **Chunk key**: `"${chunkX},${chunkY},${chunkZ}"`
- **Player chunk**: `floor(playerPos / 256)`

### Memory Layout Per Chunk
- **Voxel data**: 256³ × 4 bytes = 64MB (noise buffer)
- **Meshlet meshes**: 32³ × maxMeshSize = variable (compressed f16)
- **Density cache**: 32³ × 4 bytes = 128KB  
- **Light cache**: 32³ × 8 bytes = 256KB (RG32Float)
- **Total per chunk**: ~65-70MB

### Loading Pipeline
1. **Player movement detected** → calculate required chunk set
2. **Missing chunks identified** → add to generation queue  
3. **Queue processing** → max 2-3 chunks generating concurrently
4. **Chunk ready** → add to active set, update rendering
5. **Excess chunks** → unload oldest unused (LRU)

### Cross-Chunk Operations
- **Light propagation**: Handle boundaries with neighbor chunk communication
- **Voxel editing**: Edit operations may affect multiple chunks  
- **Seamless rendering**: Ensure no gaps between adjacent chunks

## Testing & Validation  
- [ ] **Chunk transitions**: Verify seamless loading (no pop-in/out)
- [ ] **Performance**: Maintain 60fps with 18 active chunks
- [ ] **Memory**: Stay within ~1.2GB budget for chunk data
- [ ] **Edge cases**: Rapid movement, chunk boundaries, editing
- [ ] **Multi-chunk features**: Cross-chunk lighting, seamless editing