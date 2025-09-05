import {canvas, contextUniform, device, gridSize} from "../index";
import {Block} from "./block";
import {Noise} from "./noise";
import {Mesh} from "./mesh";
import {Cull} from "./cull";
import editShader from "./voxel_edit.wgsl" with {type: "text"};
import {RenderTimer} from "./timing";

interface VoxelEditCommand {
	worldPosition: Float32Array;
	radius: number;
	operation: number; // 0 = remove, 1 = add
	timestamp: number;
}

export class VoxelEditor {
	// Dependencies
	private block: Block;
	private noise: Noise;
	private mesh: Mesh;
	private cull: Cull;

	// Position reading
	private positionReadTexture: GPUTexture;
	private positionReadBuffer: GPUBuffer;

	// Voxel editing
	private editPipeline: GPUComputePipeline;
	private editBindGroup: GPUBindGroup;
	private editUniformBindGroup: GPUBindGroup;
	private editParamsBuffer: GPUBuffer;

	// State
	private isReadingPosition = false;
	private currentWorldPosition: Float32Array | null = null;

	// Async editing state
	private editQueue: VoxelEditCommand[] = [];
	private isProcessingEdits = false;
	private pendingMeshUpdate = false;
	private timer: RenderTimer;

	constructor(block: Block, noise: Noise, mesh: Mesh, cull: Cull) {
		this.block = block;
		this.noise = noise;
		this.mesh = mesh;
		this.cull = cull;
		this.timer = new RenderTimer("voxel_editor");

		this.initPositionReading();
		this.initVoxelEditing();
	}

	private initPositionReading() {
		// Create 1x1 texture to read center pixel
		this.positionReadTexture = device.createTexture({
			size: {width: 1, height: 1},
			format: "rgba32float",
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
			label: "Voxel Edit",
			layout: "auto",
			compute: {
				module: device.createShaderModule({
					code: editShader,
				}),
				entryPoint: "main",
			},
		});

		// Create bind groups
		this.editBindGroup = device.createBindGroup({
			label: "Voxel Edit Data",
			layout: this.editPipeline.getBindGroupLayout(0),
			entries: [
				{
					binding: 0,
					resource: {buffer: this.noise.noiseBuffer},
				},
				{
					binding: 1,
					resource: {buffer: this.editParamsBuffer},
				},
			],
		});

		this.editUniformBindGroup = device.createBindGroup({
			label: "Voxel Edit Context",
			layout: this.editPipeline.getBindGroupLayout(1),
			entries: [
				{
					binding: 0,
					resource: {buffer: contextUniform.uniformBuffer},
				},
			],
		});
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
			const encoder = device.createCommandEncoder({label: "Position Read"});

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
			const mappedData = new Float32Array(this.positionReadBuffer.getMappedRange());
			this.currentWorldPosition = new Float32Array(mappedData);
			this.positionReadBuffer.unmap();

			return this.currentWorldPosition;

		} catch (error) {
			console.error("Error reading position:", error);
			return null;
		} finally {
			this.isReadingPosition = false;
		}
	}

	/**
	 * Queue voxels to be added in a sphere around the target position (non-blocking)
	 */
	addVoxels(worldPosition: Float32Array, radius: number = 2.0) {
		this.queueEdit(worldPosition, radius, 1); // 1 = add operation
	}

	/**
	 * Queue voxels to be removed in a sphere around the target position (non-blocking)
	 */
	removeVoxels(worldPosition: Float32Array, radius: number = 2.0) {
		this.queueEdit(worldPosition, radius, 0); // 0 = remove operation
	}

	/**
	 * Queue an edit command for async processing
	 */
	private queueEdit(worldPosition: Float32Array, radius: number, operation: number) {
		const command: VoxelEditCommand = {
			worldPosition: new Float32Array(worldPosition),
			radius,
			operation,
			timestamp: performance.now()
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

		// Execute all commands without blocking
		commandsToProcess.forEach(command => {
			this.executeEditCommand(command);
		});

		// Schedule mesh regeneration if any edits were processed
		if (!this.pendingMeshUpdate) {
			this.scheduleAsyncMeshUpdate();
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
		params[3] = command.radius;           // radius
		params[4] = command.operation;        // operation (0=remove, 1=add)
		// params[5-7] are padding

		device.queue.writeBuffer(this.editParamsBuffer, 0, params);

		// Run edit compute shader
		const encoder = device.createCommandEncoder({label: "Voxel Edit"});
		const computePass = encoder.beginComputePass({
			timestampWrites: this.timer.getTimestampWrites(),
		});

		computePass.setPipeline(this.editPipeline);
		computePass.setBindGroup(0, this.editBindGroup);
		computePass.setBindGroup(1, this.editUniformBindGroup);

		// Dispatch to cover all voxels (could be optimized to only affect nearby voxels)
		const workgroupsPerDim = Math.ceil(gridSize / 4);
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
	private scheduleAsyncMeshUpdate() {
		if (this.pendingMeshUpdate) return;

		this.pendingMeshUpdate = true;

		// Use a microtask to defer the mesh update without blocking
		queueMicrotask(() => {
			this.regenerateMeshesAsync();
			this.pendingMeshUpdate = false;
		});
	}

	/**
	 * Regenerate meshes asynchronously
	 */
	private regenerateMeshesAsync() {
		// Generate new meshes from modified voxel data
		const encoder = device.createCommandEncoder({label: "Async Mesh Regeneration"});

		// Update mesh generation
		this.mesh.update(encoder);

		device.queue.submit([encoder.finish()]);
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
	 * Get render time for voxel editing operations
	 */
	get renderTime(): number {
		return this.timer.renderTime;
	}

	/**
	 * Cleanup resources
	 */
	destroy() {
		this.positionReadTexture?.destroy();
		this.positionReadBuffer?.destroy();
		this.editParamsBuffer?.destroy();
	}
}