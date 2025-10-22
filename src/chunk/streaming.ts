import {Chunk} from './chunk';
import {camera, device, gpu, gridSize, scheduler} from '../index';
import {Cull} from '../pipeline/rendering/cull';
import {Mesh} from '../pipeline/generation/mesh';
import {Block} from '../pipeline/rendering/block';
import {Light} from '../pipeline/rendering/light';
import {Noise} from '../pipeline/generation/noise';
import {VoxelEditorHandler} from '../ui/voxel-editor';
import {VoxelEditor} from '../pipeline/generation/voxel_editor';

type GenerationStage = 'create' | 'noise' | 'mesh' | 'light' | 'finalize';

interface GenerationTask {
	index: number;
	position: number[];
	stage: GenerationStage;
	chunk?: Chunk;
	progress: number;
	threadAccumulator?: number; // Accumulated threads for noise generation
}

export class Streaming {
	grid = new Map<number, Chunk>();
	generationQueue: number[][] = []; // Queue of chunk positions to generate
	pendingGenerations: GenerationTask[] = [];
	queuedChunks = new Set<number>();
	inProgressGenerations = new Set<number>();
	maxConcurrentGenerations = 1;

	renderDistance = 2.5;
	pendingCleanup: Chunk[] = [];

	noise = new Noise();
	cull = new Cull();
	light = new Light();
	block = new Block();
	mesh = new Mesh();

	voxelEditor = new VoxelEditor(this.block, this.mesh, this.light, this.cull);
	voxelEditorHandler = new VoxelEditorHandler(gpu, this.voxelEditor);

	nextChunkId = 1;
	activeChunks = new Set<Chunk>();

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

		for (let dx = -1; dx <= 1; dx++) {
			for (let dy = -1; dy <= 1; dy++) {
				for (let dz = -1; dz <= 1; dz++) {
					if (dx === 0 && dy === 0 && dz === 0) {
						continue;
					}

					const neighborPos = [
						chunk.position[0] + dx,
						chunk.position[1] + dy,
						chunk.position[2] + dz,
					];
					const neighbor = this.getChunkAt(neighborPos);
					if (neighbor) {
						neighbors.push(neighbor);
					}
				}
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

	queueChunksAroundPosition(centerPos: number[]) {
		const searchRadius = Math.ceil(this.renderDistance); // Integer radius to search

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
				if (distance <= this.renderDistance) {
					const chunkIndex = this.map3D1D(chunkPos);

					// Only queue if chunk isn't present or already scheduled
					if (
						!this.grid.has(chunkIndex) &&
						!this.inProgressGenerations.has(chunkIndex) &&
						!this.queuedChunks.has(chunkIndex)
					) {
						this.generationQueue.push(chunkPos);
						this.queuedChunks.add(chunkIndex);
					}
				}
			}
		}
	}

	processGenerationQueue(): void {
		this.fillGenerationTasks();
		if (this.pendingGenerations.length === 0) {
			return;
		}

		const currentTask = this.pendingGenerations[0];

		// Process generation task without awaiting (non-blocking)
		this.advanceGeneration(currentTask).then((completed) => {
			if (completed) {
				// Remove the completed task
				const index = this.pendingGenerations.indexOf(currentTask);
				if (index !== -1) {
					this.pendingGenerations.splice(index, 1);
				}
			} else {
				// Move task to end of queue if not completed
				const index = this.pendingGenerations.indexOf(currentTask);
				if (index === 0) {
					this.pendingGenerations.shift();
					this.pendingGenerations.push(currentTask);
				}
			}
		}).catch((error) => {
			console.error('Error in chunk generation:', error);
			const index = this.pendingGenerations.indexOf(currentTask);
			if (index !== -1) {
				this.pendingGenerations.splice(index, 1);
			}
			if (currentTask.chunk) {
				this.inProgressGenerations.delete(currentTask.index);
			}
		});
	}

	update(updateEncoder: GPUCommandEncoder) {
		const center = this.cameraPositionInGridSpace;
		const octant = this.cameraOctantInChunk;

		// Process one chunk from generation queue per frame (non-blocking)
		this.processGenerationQueue();

		// Update active chunks based on new position
		this.updateActiveChunks(center);

		this.cull.setPlayerChunk(center);

		// Queue new chunks that need to be generated
		this.queueChunksAroundPosition(center);

		// Update all active chunks
		const chunksArray = Array.from(this.activeChunks);

		for (const chunk of chunksArray) {
			this.light.update(updateEncoder, chunk, (c) => this.getNeighborChunks(c));
			this.cull.update(updateEncoder, chunk, (c) => this.getNeighborChunks(c));
		}

		// Render all chunks at once
		this.block.update(updateEncoder, chunksArray);
	}

	updateActiveChunks(center: number[]) {
		this.activeChunks.clear();

		const searchRadius = Math.ceil(this.renderDistance); // Integer radius to search

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
				if (distance <= this.renderDistance) {
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
		// Do this asynchronously without blocking the next frame
		if (this.pendingCleanup.length > 0) {
			const chunksToCleanup = this.pendingCleanup;
			this.pendingCleanup = [];
			// Fire and forget - don't await
			this.cleanupChunks(chunksToCleanup).catch((error) => {
				console.warn('Error during chunk cleanup:', error);
			});
		}
	}

	private initializeChunk(position: number[]): Chunk {
		const chunkId = this.nextChunkId++;
		return new Chunk(chunkId, position);
	}

	private registerChunk(chunk: Chunk) {
		this.noise.registerChunk(chunk);
		this.mesh.registerChunk(chunk);
		this.cull.registerChunk(chunk);
		this.light.registerChunk(chunk);
		this.block.registerChunk(chunk);
		this.voxelEditor.registerChunk(chunk);
	}

	private fillGenerationTasks() {
		// Sort generation queue by distance to camera and view direction (closest and in-view first)
		if (this.generationQueue.length > 0) {
			const camX = camera.position[0];
			const camZ = camera.position[2];
			// Calculate view direction from yaw
			const viewDirX = Math.sin(camera.yaw);
			const viewDirZ = Math.cos(camera.yaw);

			this.generationQueue.sort((a, b) => {
				const aCenterX = (a[0] + 0.5) * gridSize;
				const aCenterZ = (a[2] + 0.5) * gridSize;
				const aVecX = aCenterX - camX;
				const aVecZ = aCenterZ - camZ;
				const aDist = Math.sqrt(aVecX * aVecX + aVecZ * aVecZ);
				const aDot = (aVecX * viewDirX + aVecZ * viewDirZ) / (aDist || 1);

				const bCenterX = (b[0] + 0.5) * gridSize;
				const bCenterZ = (b[2] + 0.5) * gridSize;
				const bVecX = bCenterX - camX;
				const bVecZ = bCenterZ - camZ;
				const bDist = Math.sqrt(bVecX * bVecX + bVecZ * bVecZ);
				const bDot = (bVecX * viewDirX + bVecZ * viewDirZ) / (bDist || 1);

				// Prioritize chunks in view direction (higher dot product)
				// then by distance (closer is better)
				const aScore = aDot - aDist * 0.1;
				const bScore = bDot - bDist * 0.1;

				return bScore - aScore;
			});
		}

		while (
			this.pendingGenerations.length < this.maxConcurrentGenerations &&
			this.generationQueue.length > 0
			) {
			const position = this.generationQueue.shift()!;
			const chunkIndex = this.map3D1D(position);
			this.queuedChunks.delete(chunkIndex);

			if (this.grid.has(chunkIndex) || this.inProgressGenerations.has(chunkIndex)) {
				continue;
			}

			this.inProgressGenerations.add(chunkIndex);
			this.pendingGenerations.push({
				index: chunkIndex,
				position: [...position],
				stage: 'create',
				progress: 0,
				threadAccumulator: 0,
			});
		}
	}

	private async advanceGeneration(task: GenerationTask): Promise<boolean> {
		switch (task.stage) {
			case 'create': {
				const chunk = this.initializeChunk(task.position);
				task.chunk = chunk;
				console.log('Generating chunk:', task.position);
				task.stage = 'noise';
				task.progress = 0;
				return false;
			}
			case 'noise': {
				const chunk = task.chunk;
				if (!chunk) {
					throw new Error('Chunk missing for noise stage');
				}


				if (task.progress === 0) {
					task.progress = 1;

					scheduler.work("noise_for_chunk", [chunk.position[0], chunk.position[1], chunk.position[2], gridSize]).then(res => {
						chunk.setColors(res.colors as Uint32Array);
						chunk.setNormals(res.normals as Float32Array);
						chunk.setVertices(res.vertices as Float32Array);
						chunk.setMaterialColors(res.material_colors as Uint32Array);
						chunk.setCommands(res.commands as Uint32Array);
						chunk.setDensities(res.densities as Uint32Array);
						chunk.setVertexCounts(res.vertex_counts as Uint32Array);

						this.registerChunk(chunk);

						task.stage = 'light';
						task.progress = 0;
						task.threadAccumulator = 0;
					});
				}

				return false;
			}

			case 'light': {
				const chunk = task.chunk;
				if (!chunk) {
					throw new Error('Chunk missing for light stage');
				}
				const encoder = device.createCommandEncoder();
				// Run multiple light propagation iterations to fully stabilize lighting
				for (let i = 0; i < 32; i++) {
					this.light.update(encoder, chunk, (c) => this.getNeighborChunks(c));
				}
				this.light.bakeVertexLighting(encoder, chunk, (c) => this.getNeighborChunks(c));
				device.queue.submit([encoder.finish()]);
				task.stage = 'finalize';
				return false;
			}
			case 'finalize': {
				const chunk = task.chunk;
				if (!chunk) {
					throw new Error('Chunk missing for finalize stage');
				}
				// Complete the mesh finalization (now won't block since readback is ready)
				const result = await this.mesh.completeMeshFinalization();
				if (result && result.buffersResized) {
					// Buffers were resized, need to recreate bind groups
					this.block.unregisterChunk(chunk);
					this.light.unregisterChunk(chunk);
					this.block.registerChunk(chunk);
					this.light.registerChunk(chunk);
				}
				this.grid.set(task.index, chunk);
				const neighbors = this.getNeighborChunks(chunk);
				this.cull.invalidateChunkAndNeighbors(chunk, neighbors);
				for (const neighbor of neighbors) {
					this.light.invalidate(neighbor);
				}
				this.inProgressGenerations.delete(task.index);
				return true;
			}
			default:
				return false;
		}
	}

	private async cleanupChunks(chunks: Chunk[]) {

		for (const chunk of chunks) {
			const neighbors = this.getNeighborChunks(chunk);
			const chunkIndex = this.map3D1D(chunk.position);
			this.grid.delete(chunkIndex);

			// Unregister from all pipelines (cull is async)
			this.noise.unregisterChunk(chunk);
			this.mesh.unregisterChunk(chunk);
			await this.cull.unregisterChunk(chunk);
			this.cull.invalidateChunks(neighbors);
			this.light.unregisterChunk(chunk);
			this.block.unregisterChunk(chunk);
			this.voxelEditor.unregisterChunk(chunk);

			// Destroy GPU resources
			chunk.destroy();
		}
	}
}
