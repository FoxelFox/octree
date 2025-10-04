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
	isGenerating = false;
	pendingCleanup: Chunk[] = [];

	noise = new Noise();
	light = new Light();
	block = new Block();
	mesh = new Mesh();
	cull = new Cull();

	voxelEditor = new VoxelEditor(this.block, this.mesh, this.light);
	voxelEditorHandler = new VoxelEditorHandler(gpu, this.voxelEditor);

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

		// Set up neighbor chunk getters for cross-chunk lighting
		this.voxelEditor.setNeighborChunkGetter((chunk) => this.getNeighborChunks(chunk));

		// Set up chunk getter for voxel editor
		this.voxelEditorHandler.setChunkGetter((position) => this.getChunkAt(position));

		// Calculate player's initial chunk position
		const playerChunkPos = this.cameraPositionInGridSpace;


		// Initialize active chunks with the current position
		this.updateActiveChunks(playerChunkPos);

		// Queue surrounding chunks for async generation
		this.queueChunksAroundPosition(playerChunkPos);
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
		// Bake lighting immediately to show initial results (will be re-baked when light stabilizes)
		this.light.bakeVertexLighting(meshEncoder, newChunk, (c) => this.getNeighborChunks(c));
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
		const renderDistance = 0.8; // Same as updateActiveChunks
		const searchRadius = Math.ceil(renderDistance); // Integer radius to search

		// Camera position in world space
		const camX = camera.position[0];
		const camZ = camera.position[2];

		// Load chunks within renderDistance (horizontal only)
		for (let dx = -searchRadius; dx <= searchRadius; dx++) {
			for (let dz = -searchRadius; dz <= searchRadius; dz++) {
				const chunkPos = [
					centerPos[0] + dx,
					0, // Fixed height at 0
					centerPos[2] + dz,
				];

				// Calculate distance from camera to chunk center in world space
				const chunkCenterX = (chunkPos[0] + 0.5) * gridSize;
				const chunkCenterZ = (chunkPos[2] + 0.5) * gridSize;
				const distance = Math.sqrt(
					(camX - chunkCenterX) ** 2 +
					(camZ - chunkCenterZ) ** 2
				) / gridSize;

				// Only queue chunks within circular distance
				if (distance <= renderDistance) {
					const chunkIndex = this.map3D1D(chunkPos);

					// Only queue if chunk doesn't exist
					if (!this.grid.has(chunkIndex)) {
						this.generationQueue.push(chunkPos);
					}
				}
			}
		}
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

		// Update active chunks based on new position
		this.updateActiveChunks(center);

		// Queue new chunks that need to be generated
		this.queueChunksAroundPosition(center);

		// Update all active chunks
		const chunksArray = Array.from(this.activeChunks);

		for (const chunk of chunksArray) {
			this.light.update(updateEncoder, chunk, (c) => this.getNeighborChunks(c));
			this.cull.update(updateEncoder, chunk);
		}

		// Render all chunks at once
		this.block.update(updateEncoder, chunksArray);
	}

	updateActiveChunks(center: number[]) {
		this.activeChunks.clear();

		const renderDistance = 0.8;
		const searchRadius = Math.ceil(renderDistance); // Integer radius to search

		// Camera position in world space
		const camX = camera.position[0];
		const camZ = camera.position[2];

		// Add chunks within renderDistance (horizontal only)
		for (let dx = -searchRadius; dx <= searchRadius; dx++) {
			for (let dz = -searchRadius; dz <= searchRadius; dz++) {
				const chunkPos = [
					center[0] + dx,
					0, // Fixed height at 0
					center[2] + dz,
				];

				// Calculate distance from camera to chunk center in world space
				const chunkCenterX = (chunkPos[0] + 0.5) * gridSize;
				const chunkCenterZ = (chunkPos[2] + 0.5) * gridSize;
				const distance = Math.sqrt(
					(camX - chunkCenterX) ** 2 +
					(camZ - chunkCenterZ) ** 2
				) / gridSize;

				// Only activate chunks within circular distance
				if (distance <= renderDistance) {
					const chunkIndex = this.map3D1D(chunkPos);
					const chunk = this.grid.get(chunkIndex);

					if (chunk) {
						this.activeChunks.add(chunk);
					}
				}
			}
		}

		// Build set of chunks that are needed (active + neighbors of active)
		const neededChunks = new Set<Chunk>(this.activeChunks);
		for (const chunk of this.activeChunks) {
			const neighbors = this.getNeighborChunks(chunk);
			for (const neighbor of neighbors) {
				neededChunks.add(neighbor);
			}
		}

		// Find chunks that are neither active nor neighbors of active chunks
		const chunksToRemove: Chunk[] = [];
		for (const chunk of this.grid.values()) {
			if (!neededChunks.has(chunk)) {
				chunksToRemove.push(chunk);
			}
		}

		// Queue chunks for cleanup (will happen after GPU submit)
		if (chunksToRemove.length > 0) {
			this.pendingCleanup.push(...chunksToRemove);
		}
	}

	async afterUpdate() {
		this.block.afterUpdate();
		this.mesh.afterUpdate();
		this.cull.afterUpdate();
		this.light.afterUpdate();

		// Now that GPU commands are submitted, cleanup chunks that are no longer needed
		if (this.pendingCleanup.length > 0) {
			const chunksToCleanup = this.pendingCleanup;
			this.pendingCleanup = [];
			await this.cleanupChunks(chunksToCleanup);
		}
	}

	private async cleanupChunks(chunks: Chunk[]) {

		for (const chunk of chunks) {
			const chunkIndex = this.map3D1D(chunk.position);
			this.grid.delete(chunkIndex);

			// Unregister from all pipelines (cull is async)
			this.noise.unregisterChunk(chunk);
			this.mesh.unregisterChunk(chunk);
			await this.cull.unregisterChunk(chunk);
			this.light.unregisterChunk(chunk);
			this.block.unregisterChunk(chunk);
			this.voxelEditor.unregisterChunk(chunk);

			// Destroy GPU resources
			chunk.destroy();
		}
	}
}
