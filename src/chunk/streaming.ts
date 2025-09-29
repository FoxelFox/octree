import {Chunk} from './chunk';
import {camera, device, gpu, gridSize} from '../index';
import {Cull} from '../pipeline/rendering/cull';
import {Mesh} from '../pipeline/generation/mesh';
import {Block} from '../pipeline/rendering/block';
import {Light} from '../pipeline/rendering/light';
import {Noise} from '../pipeline/generation/noise';
import {VoxelEditorHandler} from '../ui/voxel-editor';
import {VoxelEditor} from '../pipeline/generation/voxel_editor';

export class Streaming {
	grid = new Map<number, Chunk>();
	generationQueue: number[][] = []; // Queue of chunk positions to generate
	generatedChunks = new Array<Chunk>();
	isGenerating = false;

	noise = new Noise();
	light = new Light();
	block = new Block();
	mesh = new Mesh();
	cull = new Cull();

	voxelEditor = new VoxelEditor(this.block, this.mesh, this.light);
	voxelEditorHandler = new VoxelEditorHandler(gpu, this.voxelEditor);

	chunk: Chunk;
	lastChunkPos: number[] = [0, 0, 0];
	nextChunkId = 1;
	activeChunks = new Set<Chunk>();
	lastPlayerOctant: number[] = [0, 0, 0]; // Which half of the chunk player is in (0 or 1 for each axis)

	get cameraPositionInGridSpace(): number[] {
		return [
			Math.floor(camera.position[0] / gridSize),
			Math.floor(camera.position[1] / gridSize),
			Math.floor(camera.position[2] / gridSize),
		];
	}

	get cameraOctantInChunk(): number[] {
		// Get position within current chunk (0 to gridSize)
		const chunkPos = this.cameraPositionInGridSpace;
		const localX = camera.position[0] - chunkPos[0] * gridSize;
		const localY = camera.position[1] - chunkPos[1] * gridSize;
		const localZ = camera.position[2] - chunkPos[2] * gridSize;

		// Determine which half of the chunk (0 or 1)
		return [
			localX < gridSize / 2 ? 0 : 1,
			localY < gridSize / 2 ? 0 : 1,
			localZ < gridSize / 2 ? 0 : 1,
		];
	}

	map3D1D(p: number[]): number {
		return p[0] + 16384 * (p[1] + 16384 * p[2]);
	}

	getChunkAt(position: number[]): Chunk | undefined {
		return this.grid.get(this.map3D1D(position));
	}

	getNeighborChunks(chunk: Chunk): Chunk[] {
		const neighbors: Chunk[] = [];
		const offsets = [
			[-1, 0, 0], [1, 0, 0],
			[0, -1, 0], [0, 1, 0],
			[0, 0, -1], [0, 0, 1]
		];

		for (const offset of offsets) {
			const neighborPos = [
				chunk.position[0] + offset[0],
				chunk.position[1] + offset[1],
				chunk.position[2] + offset[2]
			];
			const neighbor = this.getChunkAt(neighborPos);
			if (neighbor) {
				neighbors.push(neighbor);
			}
		}

		return neighbors;
	}

	init() {
		console.log('Running one-time octree generation and compaction...');

		// Set up neighbor chunk getters for cross-chunk lighting
		this.voxelEditor.setNeighborChunkGetter((chunk) => this.getNeighborChunks(chunk));
		this.block.setNeighborChunkGetter((chunk) => this.getNeighborChunks(chunk));

		// Calculate player's initial chunk position
		const playerChunkPos = this.cameraPositionInGridSpace;
		console.log('Player chunk position:', playerChunkPos);
		this.lastChunkPos = [...playerChunkPos];

		// Generate only the player's current chunk immediately (blocking)
		this.chunk = this.generateChunk(playerChunkPos);

		// Set initial chunk for voxel editor
		this.voxelEditorHandler.setCurrentChunk(this.chunk);

		// Initialize active chunks with the current position
		this.updateActiveChunks(playerChunkPos);

		// Queue surrounding chunks for async generation
		this.queueChunksAroundPosition(playerChunkPos);

		console.log('Initial setup complete. Chunk queue size:', this.generationQueue.length);
		console.log('Initial active chunks:', this.activeChunks.size);
	}

	generateChunk(position: number[]): Chunk {
		const chunkId = this.nextChunkId++;
		const newChunk = new Chunk(chunkId, position);

		// Register chunk with all pipelines
		this.noise.registerChunk(newChunk);
		this.mesh.registerChunk(newChunk);
		this.cull.registerChunk(newChunk);
		this.light.registerChunk(newChunk);
		this.block.registerChunk(newChunk);
		this.voxelEditor.registerChunk(newChunk);

		// Generate chunk data
		const encoder = device.createCommandEncoder();
		this.noise.update(encoder, newChunk);
		device.queue.submit([encoder.finish()]);

		const meshEncoder = device.createCommandEncoder();
		this.mesh.update(meshEncoder, newChunk);
		this.light.update(meshEncoder, newChunk, (c) => this.getNeighborChunks(c));
		device.queue.submit([meshEncoder.finish()]);

		this.mesh.afterUpdate();
		this.light.afterUpdate();

		// Add to grid
		this.grid.set(this.map3D1D(position), newChunk);

		// Invalidate neighboring chunks' lighting so light propagates across boundaries
		const neighbors = this.getNeighborChunks(newChunk);
		for (const neighbor of neighbors) {
			this.light.invalidate(neighbor);
		}

		return newChunk;
	}

	queueChunksAroundPosition(centerPos: number[]) {
		console.log('Queueing chunks around position:', centerPos);

		// Get which octant of the chunk the player is in
		const octant = this.cameraOctantInChunk;

		// Load 2×2 horizontal chunks (X and Z) at height 0 only
		for (let dx = 0; dx <= 1; dx++) {
			for (let dz = 0; dz <= 1; dz++) {
				const chunkPos = [
					centerPos[0] + (octant[0] === 0 ? dx - 1 : dx),
					0, // Fixed height at 0
					centerPos[2] + (octant[2] === 0 ? dz - 1 : dz),
				];
				const chunkIndex = this.map3D1D(chunkPos);

				// Only queue if chunk doesn't exist
				if (!this.grid.has(chunkIndex)) {
					this.generationQueue.push(chunkPos);
				}
			}
		}

		console.log('Queued chunks:', this.generationQueue.length);
	}

	processGenerationQueue(): boolean {
		if (this.isGenerating || this.generationQueue.length === 0) {
			return false;
		}

		this.isGenerating = true;

		// Generate one chunk per frame to avoid blocking
		const position = this.generationQueue.shift()!;
		const chunkIndex = this.map3D1D(position);

		let generated = false;

		// Double-check it wasn't generated by another call
		if (!this.grid.has(chunkIndex)) {
			console.log('Generating chunk:', position);
			this.generateChunk(position);
			generated = true;
		}

		this.isGenerating = false;
		return generated;
	}

	update(updateEncoder: GPUCommandEncoder) {
		const center = this.cameraPositionInGridSpace;
		const octant = this.cameraOctantInChunk;

		// Process one chunk from generation queue per frame
		const chunkWasGenerated = this.processGenerationQueue();

		// Update active chunks if a new chunk was generated
		if (chunkWasGenerated) {
			this.updateActiveChunks(center);
		}

		// Check if player moved to a new chunk OR a new octant within the same chunk
		const chunkChanged = center[0] !== this.lastChunkPos[0] ||
			center[1] !== this.lastChunkPos[1] ||
			center[2] !== this.lastChunkPos[2];

		const octantChanged = octant[0] !== this.lastPlayerOctant[0] ||
			octant[1] !== this.lastPlayerOctant[1] ||
			octant[2] !== this.lastPlayerOctant[2];

		if (chunkChanged || octantChanged) {
			if (chunkChanged) {
				console.log('Player moved from chunk', this.lastChunkPos, 'to', center);
				this.lastChunkPos = [...center];

				// Update current chunk reference
				const chunkIndex = this.map3D1D(center);
				const newChunk = this.grid.get(chunkIndex);

				if (newChunk) {
					this.chunk = newChunk;
					// Update voxel editor handler with new current chunk
					this.voxelEditorHandler.setCurrentChunk(this.chunk);
				}
			}

			if (octantChanged) {
				console.log('Player moved to octant', octant);
				this.lastPlayerOctant = [...octant];
			}

			// Update active chunks based on new position
			this.updateActiveChunks(center);

			// Queue new chunks that need to be generated
			this.queueChunksAroundPosition(center);
		}

		// Update all active chunks
		const chunksArray = Array.from(this.activeChunks);

		for (const chunk of chunksArray) {
			this.light.update(updateEncoder, chunk, (c) => this.getNeighborChunks(c));
			this.cull.update(updateEncoder, chunk);
		}

		// Render all chunks at once
		this.block.update(updateEncoder, chunksArray);

		device.queue.submit([updateEncoder.finish()]);

		this.block.afterUpdate();
		this.mesh.afterUpdate();
		this.cull.afterUpdate();
		this.light.afterUpdate();
	}

	updateActiveChunks(center: number[]) {
		this.activeChunks.clear();

		// Get which octant of the chunk the player is in
		const octant = this.cameraOctantInChunk;

		// Add 2×2 horizontal chunks (X and Z) at height 0 only
		for (let dx = 0; dx <= 1; dx++) {
			for (let dz = 0; dz <= 1; dz++) {
				const chunkPos = [
					center[0] + (octant[0] === 0 ? dx - 1 : dx),
					0, // Fixed height at 0
					center[2] + (octant[2] === 0 ? dz - 1 : dz),
				];
				const chunkIndex = this.map3D1D(chunkPos);
				const chunk = this.grid.get(chunkIndex);

				if (chunk) {
					this.activeChunks.add(chunk);
				}
			}
		}

		console.log('Active chunks:', this.activeChunks.size);
	}
}
