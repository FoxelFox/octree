import {Chunk} from './chunk';
import {camera, contextUniform, device, gpu, gridSize, scheduler} from '../index';
import {Cull} from '../pipeline/rendering/cull';
import {Block} from '../pipeline/rendering/block';
import {Light} from '../pipeline/rendering/light';
import {VoxelEditorHandler} from '../ui/voxel-editor';
import {VoxelEditor} from '../pipeline/generation/voxel_editor';
import {Frustum} from './frustum';
import {LODManager} from './lod-manager';

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
	// LOD configuration - distance thresholds in chunk units for switching between LOD levels
	// LOD 0: distance <= LOD_THRESHOLDS[0] (highest quality)
	// LOD 1: distance <= LOD_THRESHOLDS[1]
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
	renderDistance = 8;
	pendingCleanup: Chunk[] = [];
	cull = new Cull();
	light = new Light();
	block = new Block();
	voxelEditor = new VoxelEditor(this.block, this.light, this.cull);
	voxelEditorHandler = new VoxelEditorHandler(gpu, this.voxelEditor);
	nextChunkId = 1;
	activeChunks = new Set<Chunk>();
	private lodManager = new LODManager();
	private frustum = new Frustum();
	private frustumReady = false;

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

	/**
	 * Check if a chunk at the given position and LOD is already queued or generating
	 * @param position Chunk position coordinates
	 * @param lod LOD level to check
	 * @returns true if the chunk is queued or currently generating
	 */
	private isGeneratingOrQueued(position: number[], lod: number): boolean {
		const chunkKey = this.getChunkKey(position, lod);
		return this.inProgressGenerations.has(chunkKey) || this.queuedChunks.has(chunkKey);
	}

	/**
	 * Invalidate lighting for all neighbors of a chunk
	 * Called when a chunk's LOD changes to ensure neighbors resample lighting
	 * @param chunk The chunk whose neighbors should have lighting invalidated
	 */
	private invalidateNeighborLighting(chunk: Chunk): void {
		const neighbors = this.getNeighborChunks(chunk);
		for (const neighbor of neighbors) {
			this.light.invalidate(neighbor);
		}
	}

	/**
	 * Get the LOD levels of the 6 adjacent neighbors (not diagonal)
	 * Order: -X, +X, -Y, +Y, -Z, +Z
	 * @param position Chunk position coordinates
	 * @returns Array of 6 LOD levels, -1 if neighbor doesn't exist
	 */
	private getNeighborLODs(position: number[]): number[] {
		const neighborLODs = new Array(6).fill(-1);
		const neighborOffsets = [
			[-1, 0, 0], // -X
			[1, 0, 0],  // +X
			[0, -1, 0], // -Y
			[0, 1, 0],  // +Y
			[0, 0, -1], // -Z
			[0, 0, 1],  // +Z
		];

		for (let i = 0; i < 6; i++) {
			const neighborPos = [
				position[0] + neighborOffsets[i][0],
				position[1] + neighborOffsets[i][1],
				position[2] + neighborOffsets[i][2],
			];
			const neighborIndex = this.map3D1D(neighborPos);
			const neighborActiveLOD = this.activeLOD.get(neighborIndex);
			if (neighborActiveLOD !== undefined) {
				neighborLODs[i] = neighborActiveLOD;
			}
		}

		return neighborLODs;
	}

	init() {

		// Set up neighbor chunk getters for cross-chunk lighting
		this.voxelEditor.setNeighborChunkGetter((chunk) => this.getNeighborChunks(chunk));

		// Set up chunk getter for voxel editor
		this.voxelEditorHandler.setChunkGetter((position) => this.getChunkAt(position));
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

				// Only queue chunks within circular distance and in the frustum
				// Use a more generous distance check for frustum (add 1 chunk margin) to preload nearby chunks
				const inFrustum = this.isChunkInFrustum(chunkPos);
				const shouldQueue = distance <= this.renderDistance && (inFrustum || distance <= 2);

				if (shouldQueue) {
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

		// Process generation task synchronously
		const completed = this.advanceGeneration(currentTask);

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
	}

	update(updateEncoder: GPUCommandEncoder) {
		const center = this.cameraPositionInGridSpace;
		const octant = this.cameraOctantInChunk;

		// Update frustum planes for culling
		this.updateFrustum();

		// Process one chunk from generation queue per frame
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

				// Only activate chunks within circular distance and in the frustum
				if (distance <= this.renderDistance && this.isChunkInFrustum(chunkPos)) {
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


	// Check if a chunk at the given position is visible in the camera frustum
	private isChunkInFrustum(position: number[]): boolean {
		// Skip frustum culling if not ready yet (first few frames)
		if (!this.frustumReady) {
			return true;
		}

		// Calculate chunk bounds in world space
		const minX = position[0] * gridSize;
		const minY = position[1] * gridSize;
		const minZ = position[2] * gridSize;
		const maxX = (position[0] + 1) * gridSize;
		const maxY = (position[1] + 1) * gridSize;
		const maxZ = (position[2] + 1) * gridSize;

		// Test against frustum
		return this.frustum.intersectsAABB(minX, minY, minZ, maxX, maxY, maxZ);
	}

	// Update frustum planes from current view-projection matrix
	private updateFrustum() {
		// Get view-projection matrix from context uniform
		// The view-projection matrix is stored at offset 12 (after 10 floats + 2 padding)
		// and is 4x4 = 16 floats
		const vpOffset = 12 + 16 + 16 + 16 + 16; // Skip to view_projection matrix
		const vp = contextUniform.uniformArray.slice(vpOffset, vpOffset + 16);
		this.frustum.extractFromViewProjection(vp);
		this.frustumReady = true;
	}

	private initializeChunk(position: number[], forceLOD?: number): Chunk {
		const chunkId = this.nextChunkId++;
		const distance = this.lodManager.calculateChunkDistance(position);

		// Use forced LOD if provided, otherwise calculate from distance
		const lod = forceLOD !== undefined ? forceLOD : this.lodManager.calculateLODForDistance(distance);

		return new Chunk(chunkId, position, lod);
	}

	private checkAndUpdateChunkLOD() {
		// Check all loaded positions to upgrade to better LODs when ready
		for (const [posIndex, lodMap] of this.grid.entries()) {
			// Get position from any chunk in the LOD map
			const anyChunk = lodMap.values().next().value as Chunk;
			if (!anyChunk) continue;

			const position = anyChunk.position;
			const distance = this.lodManager.calculateChunkDistance(position);
			const activeLod = this.activeLOD.get(posIndex);

			if (activeLod !== undefined) {
				// Queue generation of better LODs if needed
				this.queueBetterLODsIfNeeded(position, activeLod, distance, lodMap);

				// Try to switch to ready better LODs, or downgrade if too far
				const switched = this.switchToReadyBetterLODs(posIndex, activeLod, distance, lodMap);
				if (!switched) {
					this.downgradeToWorseLODsIfNeeded(posIndex, activeLod, distance, lodMap);
				}
			}
		}
	}

	/**
	 * Queue generation of better LODs based on distance
	 * @param position Chunk position coordinates
	 * @param activeLod Currently active LOD level
	 * @param distance Distance from camera to chunk
	 * @param lodMap Map of available LOD levels for this position
	 */
	private queueBetterLODsIfNeeded(position: number[], activeLod: number, distance: number, lodMap: Map<number, Chunk>) {
		const betterLOD = this.lodManager.shouldGenerateBetterLOD(activeLod, distance);
		if (betterLOD !== null && !lodMap.has(betterLOD) && !this.isGeneratingOrQueued(position, betterLOD)) {
			this.queueChunkGeneration(position, betterLOD);
		}
	}

	/**
	 * Switch to a better LOD that's already ready
	 * @param posIndex Position index in the grid
	 * @param activeLod Currently active LOD level
	 * @param distance Distance from camera to chunk
	 * @param lodMap Map of available LOD levels for this position
	 * @returns true if switched to a better LOD, false otherwise
	 */
	private switchToReadyBetterLODs(posIndex: number, activeLod: number, distance: number, lodMap: Map<number, Chunk>): boolean {
		for (let lod = 0; lod < activeLod; lod++) {
			const lodChunk = lodMap.get(lod);
			if (lodChunk && this.isChunkReadyToRender(lodChunk)) {
				if (this.lodManager.canUpgradeToLOD(lod, distance)) {
					this.activeLOD.set(posIndex, lod);
					this.pendingCullingChunks.delete(lodChunk);
					this.invalidateNeighborLighting(lodChunk);
					return true; // Only upgrade one step at a time
				}
			}
		}
		return false;
	}

	/**
	 * Downgrade to a worse LOD when camera moves away
	 * @param posIndex Position index in the grid
	 * @param activeLod Currently active LOD level
	 * @param distance Distance from camera to chunk
	 * @param lodMap Map of available LOD levels for this position
	 */
	private downgradeToWorseLODsIfNeeded(posIndex: number, activeLod: number, distance: number, lodMap: Map<number, Chunk>) {
		if (this.lodManager.shouldUpdateLOD(activeLod, distance)) {
			const targetLOD = this.lodManager.calculateLODForDistance(distance);
			if (targetLOD > activeLod) {
				const targetChunk = lodMap.get(targetLOD);
				if (targetChunk && this.isChunkReadyToRender(targetChunk)) {
					this.activeLOD.set(posIndex, targetLOD);
					this.pendingCullingChunks.delete(targetChunk);
					this.invalidateNeighborLighting(targetChunk);
				}
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
		this.sortGenerationQueueByPriority();

		// Start noise generation for chunks (not throttled - webworkers handle their own scheduling)
		while (this.generationQueue.length > 0) {
			const queueEntry = this.generationQueue.shift()!;
			const position = [queueEntry[0], queueEntry[1], queueEntry[2]];
			const posIndex = this.map3D1D(position);
			const lodMap = this.grid.get(posIndex);

			// Determine which LOD to generate
			const lodToGenerate = this.determineLODToGenerate(queueEntry, position, lodMap);
			const chunkKey = this.getChunkKey(position, lodToGenerate);
			this.queuedChunks.delete(chunkKey);

			// Skip if this LOD already exists or is generating
			if (!this.canStartGeneration(lodToGenerate, chunkKey, posIndex, lodMap)) {
				continue;
			}

			// Start noise generation immediately (not throttled)
			this.activeNoiseGenerations.add(chunkKey);
			this.startNoiseGeneration(position, posIndex, lodToGenerate, chunkKey);
		}
	}

	/**
	 * Sort generation queue by priority (view direction and distance)
	 * Chunks in front of camera and closer are prioritized
	 */
	private sortGenerationQueueByPriority() {
		if (this.generationQueue.length === 0) return;

		const camX = camera.position[0];
		const camZ = camera.position[2];
		const viewDirX = Math.sin(camera.yaw);
		const viewDirZ = Math.cos(camera.yaw);

		this.generationQueue.sort((a, b) => {
			const aScore = this.calculateChunkPriority(a, camX, camZ, viewDirX, viewDirZ);
			const bScore = this.calculateChunkPriority(b, camX, camZ, viewDirX, viewDirZ);
			return bScore - aScore;
		});
	}

	/**
	 * Calculate priority score for a chunk position
	 * Higher score = higher priority (in view direction, closer to camera)
	 * @param position Chunk position coordinates
	 * @param camX Camera X position
	 * @param camZ Camera Z position
	 * @param viewDirX View direction X component
	 * @param viewDirZ View direction Z component
	 * @returns Priority score (higher is better)
	 */
	private calculateChunkPriority(position: number[], camX: number, camZ: number, viewDirX: number, viewDirZ: number): number {
		const centerX = (position[0] + 0.5) * gridSize;
		const centerZ = (position[2] + 0.5) * gridSize;
		const vecX = centerX - camX;
		const vecZ = centerZ - camZ;
		const dist = Math.sqrt(vecX * vecX + vecZ * vecZ);
		const dot = (vecX * viewDirX + vecZ * viewDirZ) / (dist || 1);
		const distInChunks = dist / gridSize;

		// Score = viewDirection * 3.0 - distance_in_chunks * 0.5
		// Prioritize chunks in view direction with balanced distance penalty
		return dot * 3.0 - distInChunks * 0.5;
	}

	/**
	 * Determine which LOD level to generate for a queued chunk
	 * @param queueEntry Queue entry (may contain explicit LOD as 4th element)
	 * @param position Chunk position coordinates
	 * @param lodMap Map of available LOD levels for this position
	 * @returns LOD level to generate (0-2)
	 */
	private determineLODToGenerate(queueEntry: number[], position: number[], lodMap: Map<number, Chunk> | undefined): number {
		// Explicit LOD was specified in the queue
		if (queueEntry.length === 4) {
			return queueEntry[3];
		}

		// New position: always start with LOD 2 (lowest detail, fastest)
		if (!lodMap || lodMap.size === 0) {
			return 2;
		}

		// Position has chunks but no explicit LOD specified - calculate from distance
		const distance = this.lodManager.calculateChunkDistance(position);
		return this.lodManager.calculateLODForDistance(distance);
	}

	/**
	 * Check if we can start generation for this chunk
	 * @param lod LOD level to generate
	 * @param chunkKey Unique key for position + LOD
	 * @param posIndex Position index in the grid
	 * @param lodMap Map of available LOD levels for this position
	 * @returns true if generation can start, false if already exists or generating
	 */
	private canStartGeneration(lod: number, chunkKey: number, posIndex: number, lodMap: Map<number, Chunk> | undefined): boolean {
		// This LOD already exists
		if (lodMap?.has(lod)) {
			const generatingSet = this.generatingLODs.get(posIndex);
			generatingSet?.delete(lod);
			return false;
		}

		// Already generating this specific LOD
		if (this.inProgressGenerations.has(chunkKey) || this.activeNoiseGenerations.has(chunkKey)) {
			return false;
		}

		return true;
	}

	private startNoiseGeneration(position: number[], posIndex: number, lod: number, chunkKey: number) {
		const chunk = this.initializeChunk(position, lod);

		// Query neighbor LODs for transition geometry
		// Order: -X, +X, -Y, +Y, -Z, +Z (use -1 for no neighbor)
		const neighborLODs = this.getNeighborLODs(position);

		scheduler.work("noise_for_chunk", [position[0], position[1], position[2], lod, neighborLODs]).then(res => {
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

	private advanceGeneration(task: GenerationTask): boolean {
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
				}

				// Invalidate culling for the new chunk and its neighbors
				const neighbors = this.getNeighborChunks(chunk);
				this.cull.invalidateChunkAndNeighbors(chunk, neighbors);
				this.invalidateNeighborLighting(chunk);

				// Clean up tracking
				this.inProgressGenerations.delete(task.chunkKey);

				// Progressive LOD generation: Queue next LOD based on CURRENT distance from camera
				const currentDistance = this.lodManager.calculateChunkDistance(chunk.position);
				const betterLOD = this.lodManager.shouldGenerateBetterLOD(lod, currentDistance);

				if (betterLOD !== null && !lodMap.has(betterLOD) && !this.isGeneratingOrQueued(chunk.position, betterLOD)) {
					this.queueChunkGeneration(chunk.position, betterLOD);
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
