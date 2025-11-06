import {Chunk} from './chunk';
import {camera, device, gpu, gridSize, scheduler} from '../index';
import {Cull} from '../pipeline/rendering/cull';
import {Block} from '../pipeline/rendering/block';
import {Light} from '../pipeline/rendering/light';
import {VoxelEditorHandler} from '../ui/voxel-editor';
import {VoxelEditor} from '../pipeline/generation/voxel_editor';

type GenerationStage = 'gpu_upload' | 'light' | 'finalize';

interface GenerationTask {
	index: number; // Position index
	chunkKey: number; // Position + LOD key
	position: number[];
	lod: number;
	stage: GenerationStage;
	chunk: Chunk;
	progress: number;
	noiseResult?: any; // Result from webworker noise generation
}

export class Streaming {
	// Store multiple LODs per position: position_index -> LOD_level -> Chunk
	grid = new Map<number, Map<number, Chunk>>();
	// Track which LOD is currently being rendered for each position
	activeLOD = new Map<number, number>();
	// Track which LODs are currently generating for each position
	generatingLODs = new Map<number, Set<number>>();
	// Track chunks that need culling updates (non-active LODs waiting to be ready)
	pendingCullingChunks = new Set<Chunk>();

	generationQueue: number[][] = []; // Queue of chunk positions to generate
	pendingGenerations: GenerationTask[] = []; // Throttled queue for GPU upload, light, finalize
	queuedChunks = new Set<number>();
	inProgressGenerations = new Set<number>();
	activeNoiseGenerations = new Set<number>(); // Chunks currently generating noise (not throttled)

	renderDistance = 4;
	pendingCleanup: Chunk[] = [];

	cull = new Cull();
	light = new Light();
	block = new Block();

	voxelEditor = new VoxelEditor(this.block, this.light, this.cull);
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

	// Helper to create a unique key for position + LOD
	getChunkKey(position: number[], lod: number): number {
		// Use high bits for LOD (max LOD is 2, so we need 2 bits)
		return this.map3D1D(position) | (lod << 30);
	}

	// Get the active (currently rendered) chunk at a position
	getChunkAt(position: number[]): Chunk | undefined {
		const index = this.map3D1D(position);
		const activeLod = this.activeLOD.get(index);
		if (activeLod === undefined) return undefined;

		const lodMap = this.grid.get(index);
		return lodMap?.get(activeLod);
	}

	// Get a specific LOD chunk at a position
	getChunkAtLOD(position: number[], lod: number): Chunk | undefined {
		const index = this.map3D1D(position);
		const lodMap = this.grid.get(index);
		return lodMap?.get(lod);
	}

	// Check if a chunk is ready to render (has culling data)
	isChunkReadyToRender(chunk: Chunk): boolean {
		return chunk.indicesArray !== undefined && chunk.indicesArray.length > 0;
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

					// Check if this position has any LOD chunks
					const hasAnyLOD = this.grid.has(chunkIndex) && this.grid.get(chunkIndex)!.size > 0;

					// For new positions, we'll start with LOD 2, so check if LOD 2 is queued/generating
					const lod2Key = this.getChunkKey(chunkPos, 2);

					// Only queue if position has no chunks and LOD 2 isn't already scheduled
					if (
						!hasAnyLOD &&
						!this.inProgressGenerations.has(lod2Key) &&
						!this.queuedChunks.has(lod2Key)
					) {
						this.generationQueue.push(chunkPos);
						this.queuedChunks.add(lod2Key);
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

		// Prioritize GPU uploads first (they're the bottleneck)
		let currentTask = this.pendingGenerations.find(t => t.stage === 'gpu_upload');
		if (!currentTask) {
			// If no GPU uploads pending, process next task (light or finalize)
			currentTask = this.pendingGenerations[0];
		}

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
				if (index !== -1) {
					this.pendingGenerations.splice(index, 1);
					this.pendingGenerations.push(currentTask);
				}
			}
		}).catch((error) => {
			console.error('Error in chunk generation:', error);
			const index = this.pendingGenerations.indexOf(currentTask);
			if (index !== -1) {
				this.pendingGenerations.splice(index, 1);
			}
			this.inProgressGenerations.delete(currentTask.index);
		});
	}

	update(updateEncoder: GPUCommandEncoder) {
		const center = this.cameraPositionInGridSpace;
		const octant = this.cameraOctantInChunk;

		// Process one chunk from generation queue per frame (non-blocking)
		this.processGenerationQueue();

		// Check and update LOD for existing chunks
		this.checkAndUpdateChunkLOD();

		// Update active chunks based on new position
		this.updateActiveChunks(center);

		this.cull.setPlayerChunk(center);

		// Queue new chunks that need to be generated
		this.queueChunksAroundPosition(center);

		// Combine active chunks with pending culling chunks for updates
		const allChunksForUpdate = new Set([...this.activeChunks, ...this.pendingCullingChunks]);
		const chunksArray = Array.from(allChunksForUpdate);

		for (const chunk of chunksArray) {
			this.light.update(updateEncoder, chunk, (c) => this.getNeighborChunks(c));
			this.cull.update(updateEncoder, chunk, (c) => this.getNeighborChunks(c));
		}

		// Only render active chunks (not pending culling chunks)
		const renderChunks = Array.from(this.activeChunks);
		this.block.update(updateEncoder, renderChunks);
	}

	updateActiveChunks(center: number[]) {
		this.activeChunks.clear();

		const searchRadius = Math.ceil(this.renderDistance); // Integer radius to search

		// Camera position in world space
		const camX = camera.position[0];
		const camZ = camera.position[2];

		// Track which positions are within render distance
		const activePositions = new Set<number>();

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
					activePositions.add(chunkIndex);

					// Get the active chunk for this position
					const chunk = this.getChunkAt(chunkPos);
					if (chunk) {
						this.activeChunks.add(chunk);
					}
				}
			}
		}

		// Build set of chunks that are needed (active + neighbors of active)
		const neededChunks = new Set<Chunk>(this.activeChunks);
		const neededPositions = new Set<number>(activePositions);

		for (const chunk of this.activeChunks) {
			const neighbors = this.getNeighborChunks(chunk);
			for (const neighbor of neighbors) {
				neededChunks.add(neighbor);
				neededPositions.add(this.map3D1D(neighbor.position));
			}
		}

		// Find positions that are no longer needed and cleanup all their LOD chunks
		const positionsToRemove: number[] = [];
		for (const [posIndex, lodMap] of this.grid.entries()) {
			if (!neededPositions.has(posIndex)) {
				positionsToRemove.push(posIndex);
			}
		}

		// Queue all LODs at unneeded positions for cleanup
		for (const posIndex of positionsToRemove) {
			const lodMap = this.grid.get(posIndex);
			if (lodMap) {
				for (const chunk of lodMap.values()) {
					this.pendingCleanup.push(chunk);
				}
			}
		}
	}

	async afterUpdate() {
		this.cull.afterUpdate();

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

	private calculateLODForDistance(distance: number): number {
		// LOD levels: 0 = full resolution, 1 = half, 2 = quarter
		// Distance thresholds in chunk units
		const LOD_THRESHOLDS = [0.5, 1.5];

		for (let lod = 0; lod < LOD_THRESHOLDS.length; lod++) {
			if (distance <= LOD_THRESHOLDS[lod]) {
				return lod;
			}
		}
		return LOD_THRESHOLDS.length;
	}

	private shouldUpdateLOD(currentLOD: number, distance: number): boolean {
		const HYSTERESIS = 0.2;
		const LOD_THRESHOLDS = [0.5, 1.5];

		// Calculate target LOD with hysteresis to prevent rapid switching
		// Hysteresis makes it harder to change LOD - we need to be clearly past the threshold
		let targetLOD = 0;
		for (let lod = 0; lod < LOD_THRESHOLDS.length; lod++) {
			const threshold = LOD_THRESHOLDS[lod];

			// Apply hysteresis based on where we currently are
			// If we're at a higher LOD (lod+1 or more), need to get clearly below threshold to upgrade
			// If we're at a lower LOD (lod or less), need to get clearly above threshold to downgrade
			let adjustedThreshold: number;
			if (currentLOD > lod) {
				// Currently at higher LOD than lod, need distance < threshold - hysteresis to upgrade
				adjustedThreshold = threshold - HYSTERESIS;
			} else {
				// Currently at lower LOD than lod+1, need distance > threshold + hysteresis to downgrade
				adjustedThreshold = threshold + HYSTERESIS;
			}

			if (distance > adjustedThreshold) {
				targetLOD = lod + 1;
			} else {
				break;
			}
		}

		return targetLOD !== currentLOD;
	}

	private calculateChunkDistance(position: number[]): number {
		const chunkCenterX = (position[0] + 0.5) * gridSize;
		const chunkCenterY = (position[1] + 0.5) * gridSize;
		const chunkCenterZ = (position[2] + 0.5) * gridSize;

		const dx = camera.position[0] - chunkCenterX;
		const dy = camera.position[1] - chunkCenterY;
		const dz = camera.position[2] - chunkCenterZ;
		return Math.sqrt(dx * dx + dy * dy + dz * dz) / gridSize;
	}

	private initializeChunk(position: number[], forceLOD?: number): Chunk {
		const chunkId = this.nextChunkId++;
		const distance = this.calculateChunkDistance(position);

		// Use forced LOD if provided, otherwise calculate from distance
		const lod = forceLOD !== undefined ? forceLOD : this.calculateLODForDistance(distance);

		return new Chunk(chunkId, position, lod);
	}

	private checkAndUpdateChunkLOD() {
		// Check all loaded positions to upgrade to better LODs when ready
		for (const [posIndex, lodMap] of this.grid.entries()) {
			// Get position from any chunk in the LOD map
			const anyChunk = lodMap.values().next().value as Chunk;
			if (!anyChunk) continue;

			const position = anyChunk.position;

			// Get current active LOD
			const activeLod = this.activeLOD.get(posIndex);

			// SIMPLE RULE: Always use the best (lowest) available LOD that's ready
			// NEVER downgrade - once we have better quality, keep it
			if (activeLod !== undefined) {
				// Check if there's a better LOD available (0 is best, 2 is worst)
				for (let lod = 0; lod < activeLod; lod++) {
					const lodChunk = lodMap.get(lod);
					if (lodChunk && this.isChunkReadyToRender(lodChunk)) {
						// Found a better LOD that's ready - switch to it
						console.log(`[LOD] Upgrading [${position[0]},${position[1]},${position[2]}] from LOD ${activeLod} to LOD ${lod}`);
						this.activeLOD.set(posIndex, lod);
						this.pendingCullingChunks.delete(lodChunk);
						break; // Only upgrade one step at a time to avoid jumps
					}
				}
				// Never downgrade - we keep the best LOD we have
			}
		}
	}

	// Helper to queue generation of a specific LOD for a position
	private queueChunkGeneration(position: number[], lod: number) {
		const posIndex = this.map3D1D(position);
		const chunkKey = this.getChunkKey(position, lod);

		// Track that this LOD is being generated
		if (!this.generatingLODs.has(posIndex)) {
			this.generatingLODs.set(posIndex, new Set());
		}
		this.generatingLODs.get(posIndex)!.add(lod);

		// Queue for generation - store position AND LOD
		// We encode the LOD in the position array as a 4th element
		this.generationQueue.push([position[0], position[1], position[2], lod]);
		this.queuedChunks.add(chunkKey);
	}

	private registerChunk(chunk: Chunk) {
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

		// Start noise generation for chunks (not throttled - webworkers handle their own scheduling)
		while (this.generationQueue.length > 0) {
			const queueEntry = this.generationQueue.shift()!;

			// Extract position and LOD (4th element is optional LOD)
			const position = [queueEntry[0], queueEntry[1], queueEntry[2]];
			const posIndex = this.map3D1D(position);
			const lodMap = this.grid.get(posIndex);
			const hasAnyLOD = lodMap && lodMap.size > 0;

			// Determine which LOD to generate
			let lodToGenerate: number;

			if (queueEntry.length === 4) {
				// Explicit LOD was specified in the queue
				lodToGenerate = queueEntry[3];
				console.log(`[LOD] Processing queue entry for [${position[0]},${position[1]},${position[2]}] with explicit LOD ${lodToGenerate}`);
			} else if (!hasAnyLOD) {
				// New position: always start with LOD 2 (lowest detail, fastest)
				lodToGenerate = 2;
			} else {
				// Position has chunks: determine which LOD was requested
				const targetLod = this.targetLOD.get(posIndex);
				const activeLod = this.activeLOD.get(posIndex);

				if (targetLod !== undefined && activeLod !== undefined) {
					// Generate the next LOD towards target
					if (targetLod < activeLod) {
						lodToGenerate = activeLod - 1; // Better LOD
					} else {
						lodToGenerate = targetLod; // Worse LOD
					}
				} else {
					// Fallback: calculate from distance
					const distance = this.calculateChunkDistance(position);
					lodToGenerate = this.calculateLODForDistance(distance);
				}
			}

			// Check if this specific LOD already exists or is generating
			const chunkKey = this.getChunkKey(position, lodToGenerate);
			this.queuedChunks.delete(chunkKey);

			if (lodMap?.has(lodToGenerate)) {
				// This LOD already exists, skip
				const generatingSet = this.generatingLODs.get(posIndex);
				generatingSet?.delete(lodToGenerate);
				continue;
			}

			// Skip if already generating this specific LOD
			if (this.inProgressGenerations.has(chunkKey) || this.activeNoiseGenerations.has(chunkKey)) {
				continue;
			}

			// Start noise generation immediately (not throttled)
			this.activeNoiseGenerations.add(chunkKey);
			this.startNoiseGeneration(position, posIndex, lodToGenerate, chunkKey);
		}
	}

	private startNoiseGeneration(position: number[], posIndex: number, lod: number, chunkKey: number) {
		const chunk = this.initializeChunk(position, lod);

		scheduler.work("noise_for_chunk", [position[0], position[1], position[2], lod]).then(res => {
			this.activeNoiseGenerations.delete(chunkKey);

			// Add to throttled queue for GPU upload
			if (!this.inProgressGenerations.has(chunkKey)) {
				this.inProgressGenerations.add(chunkKey);
				this.pendingGenerations.push({
					index: posIndex,
					chunkKey: chunkKey,
					position: [...position],
					lod: lod,
					stage: 'gpu_upload',
					chunk: chunk,
					progress: 0,
					noiseResult: res,
				});
			}
		}).catch((error) => {
			console.error('Error in noise generation:', error);
			this.activeNoiseGenerations.delete(chunkKey);
		});
	}

	private async advanceGeneration(task: GenerationTask): Promise<boolean> {
		switch (task.stage) {
			case 'gpu_upload': {
				const chunk = task.chunk;
				const res = task.noiseResult;

				// Upload all mesh data to GPU (throttled to 1 chunk per frame)
				chunk.setColors(res.colors as Uint32Array);
				chunk.setNormals(res.normals as Float32Array);
				chunk.setVertices(res.vertices as Float32Array);
				chunk.setMaterialColors(res.material_colors as Uint32Array);
				chunk.setCommands(res.commands as Uint32Array);
				chunk.setDensities(res.densities as Uint32Array);
				chunk.setVertexCounts(res.vertex_counts as Uint32Array);
				chunk.setIndices(res.indices as Uint32Array);

				this.registerChunk(chunk);

				task.stage = 'light';
				task.noiseResult = undefined; // Free memory
				return false;
			}

			case 'light': {
				const chunk = task.chunk;
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
				const posIndex = task.index;
				const lod = task.lod;

				// Add chunk to LOD map
				if (!this.grid.has(posIndex)) {
					this.grid.set(posIndex, new Map());
				}
				const lodMap = this.grid.get(posIndex)!;
				lodMap.set(lod, chunk);

				// Remove from generating set
				const generatingSet = this.generatingLODs.get(posIndex);
				generatingSet?.delete(lod);

				// Determine if we should switch to this LOD
				const currentActiveLod = this.activeLOD.get(posIndex);

				if (currentActiveLod === undefined) {
					// First chunk for this position: activate immediately
					// (It's LOD 2, so it should render quickly even without culling data ready)
					this.activeLOD.set(posIndex, lod);
				} else {
					// Position already has an active LOD
					// Add the new chunk to pendingCullingChunks so culling runs on it
					// This allows us to check if it's ready in checkAndUpdateChunkLOD
					this.pendingCullingChunks.add(chunk);
					console.log(`[LOD] Added LOD ${lod} chunk at [${chunk.position[0]},${chunk.position[1]},${chunk.position[2]}] to pendingCulling (active LOD is ${currentActiveLod})`);
				}

				// Invalidate culling for the new chunk and its neighbors
				const neighbors = this.getNeighborChunks(chunk);
				this.cull.invalidateChunkAndNeighbors(chunk, neighbors);
				for (const neighbor of neighbors) {
					this.light.invalidate(neighbor);
				}

				// Clean up tracking
				this.inProgressGenerations.delete(task.chunkKey);

				// Progressive LOD generation: ALWAYS queue next better LOD immediately
				if (lod === 2) {
					// Just finished LOD 2, always queue LOD 1
					const nextLOD = 1;
					const nextChunkKey = this.getChunkKey(chunk.position, nextLOD);
					console.log(`[LOD] Finished LOD 2 at [${chunk.position[0]},${chunk.position[1]},${chunk.position[2]}], queueing LOD 1...`);
					if (!lodMap.has(nextLOD) && !this.inProgressGenerations.has(nextChunkKey) && !this.queuedChunks.has(nextChunkKey)) {
						this.queueChunkGeneration(chunk.position, nextLOD);
					}
				} else if (lod === 1) {
					// Just finished LOD 1, always queue LOD 0
					const nextLOD = 0;
					const nextChunkKey = this.getChunkKey(chunk.position, nextLOD);
					console.log(`[LOD] Finished LOD 1 at [${chunk.position[0]},${chunk.position[1]},${chunk.position[2]}], queueing LOD 0...`);
					if (!lodMap.has(nextLOD) && !this.inProgressGenerations.has(nextChunkKey) && !this.queuedChunks.has(nextChunkKey)) {
						this.queueChunkGeneration(chunk.position, nextLOD);
					}
				}

				return true;
			}
			default:
				return false;
		}
	}

	private async cleanupChunks(chunks: Chunk[]) {
		for (const chunk of chunks) {
			const neighbors = this.getNeighborChunks(chunk);
			const posIndex = this.map3D1D(chunk.position);
			const lod = chunk.lod;

			// Remove chunk from its LOD map
			const lodMap = this.grid.get(posIndex);
			if (lodMap) {
				lodMap.delete(lod);

				// If this was the last LOD for this position, cleanup tracking data
				if (lodMap.size === 0) {
					this.grid.delete(posIndex);
					this.activeLOD.delete(posIndex);
					this.generatingLODs.delete(posIndex);
				} else if (this.activeLOD.get(posIndex) === lod) {
					// If we're removing the active LOD, switch to another available LOD
					// Prefer lower (better) LODs
					const availableLODs = Array.from(lodMap.keys()).sort((a, b) => a - b);
					if (availableLODs.length > 0) {
						this.activeLOD.set(posIndex, availableLODs[0]);
					}
				}
			}

			// Remove from activeChunks and pendingCullingChunks
			this.activeChunks.delete(chunk);
			this.pendingCullingChunks.delete(chunk);

			// Unregister from all pipelines (cull is async)
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
