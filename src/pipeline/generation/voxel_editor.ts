import {canvas, contextUniform, device, gridSize} from '../../index';
import editShader from './voxel_edit.wgsl' with {type: 'text'};
import {RenderTimer} from '../timing';
import {Chunk} from "../../chunk/chunk";
import {Block} from "../rendering/block";
import {Mesh} from "./mesh";
import {Light} from "../rendering/light";

interface VoxelEditCommand {
	worldPosition: Float32Array;
	radius: number;
	operation: number; // 0 = remove, 1 = add
	timestamp: number;
	chunk: Chunk;
}

interface ChangeBounds {
	min: [number, number, number];
	max: [number, number, number];
}

export class VoxelEditor {
	// Dependencies
	private block: Block;
	private mesh: Mesh;
	private light: Light;
	private getNeighborChunks?: (chunk: Chunk) => Chunk[];

	// Position reading
	private positionReadTexture: GPUTexture;
	private positionReadBuffer: GPUBuffer;

	// Voxel editing
	private editPipeline: GPUComputePipeline;
	private chunkBindGroups = new Map<Chunk, GPUBindGroup>();
	private chunkUniformBindGroups = new Map<Chunk, GPUBindGroup>();
	private chunkWorldPosBuffers = new Map<Chunk, GPUBuffer>();
	private activeChunk: Chunk | null = null;
	private editParamsBuffer: GPUBuffer;

	// State
	private isReadingPosition = false;
	private currentWorldPosition: Float32Array | null = null;

	// Async editing state
	private editQueue: VoxelEditCommand[] = [];
	private isProcessingEdits = false;
	private pendingMeshUpdate = false;
	private changeBounds: ChangeBounds | null = null;
	private timer: RenderTimer;

	constructor(block: Block, mesh: Mesh, light: Light) {
		this.block = block;
		this.mesh = mesh;
		this.light = light;
		this.timer = new RenderTimer('voxel_editor');

		this.initPositionReading();
		this.initVoxelEditing();
	}

	/**
	 * Set callback to get neighboring chunks for light invalidation
	 */
	setNeighborChunkGetter(getter: (chunk: Chunk) => Chunk[]) {
		this.getNeighborChunks = getter;
	}

	/**
	 * Get render time for voxel editing operations
	 */
	get renderTime(): number {
		return this.timer.renderTime;
	}

	/**
	 * Reads the world position at screen center from the G-buffer position texture
	 */
	async readPositionAtCenter(): Promise<Float32Array | null> {
		if (this.isReadingPosition) {
			return this.currentWorldPosition;
		}

		this.isReadingPosition = true;

		try {
			const encoder = device.createCommandEncoder({label: 'Position Read'});

			// Calculate center pixel coordinates
			const centerX = Math.floor(canvas.width / 2);
			const centerY = Math.floor(canvas.height / 2);

			// Copy center pixel from position texture to 1x1 texture
			encoder.copyTextureToTexture(
				{
					texture: this.block.positionTexture,
					origin: {x: centerX, y: centerY},
				},
				{
					texture: this.positionReadTexture,
					origin: {x: 0, y: 0},
				},
				{width: 1, height: 1}
			);

			// Copy texture data to buffer
			encoder.copyTextureToBuffer(
				{
					texture: this.positionReadTexture,
				},
				{
					buffer: this.positionReadBuffer,
					bytesPerRow: 256, // Must be multiple of 256
					rowsPerImage: 1,
				},
				{width: 1, height: 1}
			);

			device.queue.submit([encoder.finish()]);

			// Read the data
			await this.positionReadBuffer.mapAsync(GPUMapMode.READ);
			const mappedData = new Float32Array(
				this.positionReadBuffer.getMappedRange()
			);
			this.currentWorldPosition = new Float32Array(mappedData);
			this.positionReadBuffer.unmap();

			return this.currentWorldPosition;
		} catch (error) {
			console.error('Error reading position:', error);
			return null;
		} finally {
			this.isReadingPosition = false;
		}
	}

	/**
	 * Queue voxels to be added in a sphere around the target position (non-blocking)
	 */
	addVoxels(worldPosition: Float32Array, radius: number = 2.0, chunk?: Chunk) {
		this.queueEdit(worldPosition, radius, 1, chunk); // 1 = add operation
	}

	/**
	 * Queue voxels to be removed in a sphere around the target position (non-blocking)
	 */
	removeVoxels(worldPosition: Float32Array, radius: number = 2.0, chunk?: Chunk) {
		this.queueEdit(worldPosition, radius, 0, chunk); // 0 = remove operation
	}

	registerChunk(chunk: Chunk) {
		// Create bind groups
		const bindGroup = device.createBindGroup({
			label: 'Voxel Edit Data',
			layout: this.editPipeline.getBindGroupLayout(0),
			entries: [
				{
					binding: 0,
					resource: {buffer: chunk.voxelData},
				},
				{
					binding: 1,
					resource: {buffer: this.editParamsBuffer},
				},
			],
		});

		this.chunkBindGroups.set(chunk, bindGroup);

		// Create chunk world position buffer
		const chunkWorldPosBuffer = device.createBuffer({
			size: 16, // vec3<i32> + padding = 16 bytes
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		// Write chunk world position (in voxels)
		const chunkWorldPosData = new Int32Array([
			chunk.position[0] * gridSize,
			chunk.position[1] * gridSize,
			chunk.position[2] * gridSize,
			0 // padding
		]);
		device.queue.writeBuffer(chunkWorldPosBuffer, 0, chunkWorldPosData);

		// Create uniform bind group with chunk world position
		const uniformBindGroup = device.createBindGroup({
			label: 'Voxel Edit Context',
			layout: this.editPipeline.getBindGroupLayout(1),
			entries: [
				{
					binding: 0,
					resource: {buffer: contextUniform.uniformBuffer},
				},
				{
					binding: 1,
					resource: {buffer: chunkWorldPosBuffer},
				},
			],
		});

		this.chunkUniformBindGroups.set(chunk, uniformBindGroup);
		this.chunkWorldPosBuffers.set(chunk, chunkWorldPosBuffer);

		if (!this.activeChunk) {
			this.activeChunk = chunk;
		}
	}

	/**
	 * Get the current world position at screen center
	 */
	getCurrentPosition(): Float32Array | null {
		return this.currentWorldPosition;
	}

	/**
	 * Check if there's valid geometry at screen center
	 */
	hasGeometryAtCenter(): boolean {
		if (!this.currentWorldPosition) return false;

		// Check if the w component (distance) indicates valid geometry
		// In the deferred renderer, valid geometry has a meaningful distance value
		return this.currentWorldPosition[3] > 0;
	}

	/**
	 * Cleanup resources
	 */
	destroy() {
		this.positionReadTexture?.destroy();
		this.positionReadBuffer?.destroy();
		this.editParamsBuffer?.destroy();
	}

	private initPositionReading() {
		// Create 1x1 texture to read center pixel
		this.positionReadTexture = device.createTexture({
			size: {width: 1, height: 1},
			format: 'rgba32float',
			usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
		});

		// Create buffer to read position data to CPU
		// Must be a multiple of 256 bytes for WebGPU buffer alignment
		this.positionReadBuffer = device.createBuffer({
			size: 256, // Minimum required size for texture-to-buffer copy
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
		});
	}

	private initVoxelEditing() {
		// Create edit parameters buffer
		this.editParamsBuffer = device.createBuffer({
			size: 32, // vec3 position + float radius + vec3 operation + padding
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		// Create edit compute pipeline
		this.editPipeline = device.createComputePipeline({
			label: 'Voxel Edit',
			layout: 'auto',
			compute: {
				module: device.createShaderModule({
					code: editShader,
				}),
				entryPoint: 'main',
			},
		});


	}

	/**
	 * Queue an edit command for async processing
	 */
	private queueEdit(
		worldPosition: Float32Array,
		radius: number,
		operation: number,
		chunk?: Chunk
	) {
		const targetChunk = chunk ?? this.activeChunk;
		if (!targetChunk) {
			console.warn('Voxel edit requested before chunk registration.');
			return;
		}
		this.activeChunk = targetChunk;

		const command: VoxelEditCommand = {
			worldPosition: new Float32Array(worldPosition),
			radius,
			operation,
			timestamp: performance.now(),
			chunk: targetChunk,
		};
		this.editQueue.push(command);

		// Start processing if not already running
		if (!this.isProcessingEdits) {
			this.processEditQueue();
		}
	}

	/**
	 * Process the edit queue asynchronously without blocking the render loop
	 */
	private processEditQueue() {
		if (this.isProcessingEdits || this.editQueue.length === 0) {
			return;
		}

		this.isProcessingEdits = true;

		// Process all queued edits in a single batch
		const commandsToProcess = [...this.editQueue];
		this.editQueue.length = 0; // Clear the queue

		// Reset change bounds for this batch
		this.changeBounds = null;

		// Execute all commands without blocking
		const firstChunk = commandsToProcess[0]?.chunk;
		commandsToProcess.forEach((command) => {
			if (command.chunk !== firstChunk) {
				console.warn('Voxel edit commands span multiple chunks; only the first chunk will be updated.');
				return;
			}
			this.executeEditCommand(command);
			this.updateChangeBounds(command.worldPosition, command.radius);
		});

		// Schedule mesh regeneration if any edits were processed
		if (!this.pendingMeshUpdate && firstChunk) {
			this.scheduleAsyncMeshUpdate(firstChunk);
		}

		this.isProcessingEdits = false;
	}

	/**
	 * Execute a single edit command
	 */
	private executeEditCommand(command: VoxelEditCommand) {
		// Update edit parameters
		const params = new Float32Array(8); // 8 floats to align to 32 bytes
		params[0] = command.worldPosition[0]; // position.x
		params[1] = command.worldPosition[1]; // position.y
		params[2] = command.worldPosition[2]; // position.z
		params[3] = command.radius; // radius
		params[4] = command.operation; // operation (0=remove, 1=add)
		// params[5-7] are padding

		device.queue.writeBuffer(this.editParamsBuffer, 0, params);

		// Run edit compute shader
		const bindGroup = this.chunkBindGroups.get(command.chunk);
		if (!bindGroup) {
			console.warn('No voxel edit bind group for chunk', command.chunk.id);
			return;
		}

		const encoder = device.createCommandEncoder({label: 'Voxel Edit'});
		const computePass = encoder.beginComputePass({
			timestampWrites: this.timer.getTimestampWrites(),
		});

		computePass.setPipeline(this.editPipeline);
		computePass.setBindGroup(0, bindGroup);
		computePass.setBindGroup(1, this.chunkUniformBindGroups.get(command.chunk)!);

		// Dispatch to cover all voxels including border (257³)
		const voxelGridSize = gridSize + 1;
		const workgroupsPerDim = Math.ceil(voxelGridSize / 4);
		computePass.dispatchWorkgroups(
			workgroupsPerDim,
			workgroupsPerDim,
			workgroupsPerDim
		);

		computePass.end();
		this.timer.resolveTimestamps(encoder);
		device.queue.submit([encoder.finish()]);

		// Read timing after submission
		this.timer.readTimestamps();
	}

	/**
	 * Schedule async mesh regeneration that doesn't block the render loop
	 */
	private scheduleAsyncMeshUpdate(chunk: Chunk) {
		if (this.pendingMeshUpdate) return;

		this.pendingMeshUpdate = true;

		// Use a microtask to defer the mesh update without blocking
		queueMicrotask(() => {
			this.regenerateMeshesAsync(chunk);
			this.pendingMeshUpdate = false;
		});
	}

	/**
	 * Update the bounds of changed areas
	 */
	private updateChangeBounds(worldPosition: Float32Array, radius: number) {
		const minBound = [
			worldPosition[0] - radius,
			worldPosition[1] - radius,
			worldPosition[2] - radius,
		] as [number, number, number];
		const maxBound = [
			worldPosition[0] + radius,
			worldPosition[1] + radius,
			worldPosition[2] + radius,
		] as [number, number, number];

		if (!this.changeBounds) {
			this.changeBounds = {min: minBound, max: maxBound};
		} else {
			// Expand existing bounds
			for (let i = 0; i < 3; i++) {
				this.changeBounds.min[i] = Math.min(
					this.changeBounds.min[i],
					minBound[i]
				);
				this.changeBounds.max[i] = Math.max(
					this.changeBounds.max[i],
					maxBound[i]
				);
			}
		}
	}

	/**
	 * Regenerate meshes asynchronously
	 */
	private regenerateMeshesAsync(chunk: Chunk) {
		// Generate new meshes from modified voxel data
		const encoder = device.createCommandEncoder({
			label: 'Async Mesh Regeneration',
		});

		// Convert world-space bounds to chunk-local bounds
		let localBounds: { min: [number, number, number]; max: [number, number, number] } | undefined;
		if (this.changeBounds) {
			const chunkWorldPos = [
				chunk.position[0] * gridSize,
				chunk.position[1] * gridSize,
				chunk.position[2] * gridSize
			];

			// Convert world-space to chunk-local coordinates
			localBounds = {
				min: [
					Math.max(0, this.changeBounds.min[0] - chunkWorldPos[0]),
					Math.max(0, this.changeBounds.min[1] - chunkWorldPos[1]),
					Math.max(0, this.changeBounds.min[2] - chunkWorldPos[2])
				],
				max: [
					Math.min(gridSize - 1, this.changeBounds.max[0] - chunkWorldPos[0]),
					Math.min(gridSize - 1, this.changeBounds.max[1] - chunkWorldPos[1]),
					Math.min(gridSize - 1, this.changeBounds.max[2] - chunkWorldPos[2])
				]
			};

			// Validate bounds - if invalid, regenerate entire chunk
			if (localBounds.min[0] > localBounds.max[0] ||
				localBounds.min[1] > localBounds.max[1] ||
				localBounds.min[2] > localBounds.max[2]) {
				console.log('Invalid local bounds, regenerating entire chunk');
				localBounds = undefined;
			}
		}

		// Update mesh generation with chunk-local bounds
		this.mesh.update(encoder, chunk, localBounds);

		device.queue.submit([encoder.finish()]);

		// Invalidate lighting after voxel changes
		this.light.invalidate(chunk);

		// Also invalidate neighboring chunks' lighting so light propagates across boundaries
		if (this.getNeighborChunks) {
			const neighbors = this.getNeighborChunks(chunk);
			for (const neighbor of neighbors) {
				this.light.invalidate(neighbor);
			}
		}
	}
}
