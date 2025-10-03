# Chunk Streaming Optimization TODO

## Critical Performance Issues

### 1. Command Encoder Synchronization (index.ts:67, streaming.ts:262)
**Current**: Creates encoder in loop, immediately submits in `streaming.update()`
- Forces synchronization every frame
- Prevents GPU work batching
- **Fix**: Pass encoder to `streaming.update()` and let caller control submission

### 2. Synchronous Chunk Generation (streaming.ts:115-149)
**Current**: `generateChunk()` blocks on each generation stage
```typescript
// Line 128-130: Blocks on noise
const encoder = device.createCommandEncoder();
this.noise.update(encoder, newChunk);
device.queue.submit([encoder.finish()]);

// Line 132-135: Then blocks on mesh/light
const meshEncoder = device.createCommandEncoder();
this.mesh.update(meshEncoder, newChunk);
this.light.update(meshEncoder, newChunk, ...);
device.queue.submit([meshEncoder.finish()]);
```
- **Fix**: Make async, pipeline stages, batch multiple chunks

### 3. Excessive Array Conversions (streaming.ts:252)
**Current**: `Array.from(this.activeChunks)` every frame
- **Fix**: Cache array or iterate Set directly

### 4. Inefficient Neighbor Lookup (streaming.ts:64-85)
**Current**: Creates arrays on every call
```typescript
const offsets = [
    [-1, 0, 0], [1, 0, 0],
    [0, -1, 0], [0, 1, 0],
    [0, 0, -1], [0, 0, 1]
];
```
- Called frequently (lines 255, 304)
- **Fix**: Static cached offset array

### 5. Single Chunk Per Frame (streaming.ts:178-200)
**Current**: `processGenerationQueue()` generates 1 chunk/frame with flag
- Causes visible pop-in on fast movement
- **Fix**: Batch multiple chunks or async generation with priority queue

### 6. Blocking Cleanup (streaming.ts:326-348)
**Current**: `await device.queue.onSubmittedWorkDone()` stalls cleanup
- All unregistrations serial
- **Fix**: Batch cleanup or use timestamp queries

### 7. Underutilized Double Buffering (chunk.ts:86-96)
**Current**: Light buffers double buffered but not async swapped
- **Fix**: Overlap compute with rendering

## Medium Priority Issues

### 8. Memory Allocations in Hot Path
- Line 163: `[centerPos[0] + ..., 0, centerPos[2] + ...]`
- Line 230: `[...center]`
- Line 241: `[...octant]`
- **Fix**: Reuse objects, mutate in place

### 9. Grid Lookup Hash Function (streaming.ts:56-58)
**Current**: `p[0] + 16384 * (p[1] + 16384 * p[2])`
- 2 multiplications per lookup
- Called multiple times/frame for same positions
- **Fix**: Cache recent lookups (LRU map)

### 10. Console Logging in Production (multiple locations)
- Lines 88, 96, 111, 112, 153, 175, 193, 225, 240, 323, 347
- **Fix**: Debug flag or remove from hot paths

## Implementation Priority

1. **Remove console.logs** (quick win, easy)
2. **Cache neighbor offsets** (quick win, easy)
3. **Cache activeChunks array** (quick win, easy)
4. **Batch encoder submissions** (medium, high impact)
5. **Async chunk generation** (hard, high impact)
6. **Multi-chunk per frame** (medium, high impact)
7. **Optimize cleanup** (medium, medium impact)
8. **Cache position arrays** (easy, low impact)
9. **Cache hash lookups** (medium, low impact)
10. **Better double buffering** (hard, medium impact)

## Detailed Fixes

### Fix #1: Batch Encoder Submissions
```typescript
// index.ts
const updateEncoder = device.createCommandEncoder();
streaming.update(updateEncoder);
// Don't submit here - let streaming.update() batch everything

// streaming.ts:262
// Only submit once at end after all pipelines updated
```

### Fix #2: Async Chunk Generation
```typescript
async generateChunkAsync(position: number[]): Promise<Chunk> {
    // Register all pipelines
    // Submit all generation work in single encoder
    // Return chunk immediately, mark as "generating"
    // Use fence/timestamp to know when ready
}
```

### Fix #3: Static Neighbor Offsets
```typescript
private static readonly NEIGHBOR_OFFSETS = [
    [-1, 0, 0], [1, 0, 0],
    [0, -1, 0], [0, 1, 0],
    [0, 0, -1], [0, 0, 1]
] as const;
```

### Fix #4: Cache Active Chunks Array
```typescript
private activeChunksArray: Chunk[] = [];
private activeChunksDirty = true;

updateActiveChunks(center: number[]) {
    // ... existing logic ...
    this.activeChunksDirty = true;
}

update() {
    if (this.activeChunksDirty) {
        this.activeChunksArray = Array.from(this.activeChunks);
        this.activeChunksDirty = false;
    }
    // Use this.activeChunksArray
}
```

### Fix #5: Multi-Chunk Processing
```typescript
processGenerationQueue(budget: number = 2): number {
    let generated = 0;
    while (generated < budget && this.generationQueue.length > 0) {
        // Generate chunk
        generated++;
    }
    return generated;
}
```
