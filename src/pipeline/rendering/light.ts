import {compression, contextUniform, device, gridSize} from "../../index";
import shader from "./light.wgsl" with {type: "text"};
import {RenderTimer} from "../timing";
import {Chunk} from "../../chunk/chunk";

export class Light {

	// Compute pipeline for light propagation
	pipeline: GPUComputePipeline;
	bindGroups = new Map<Chunk, GPUBindGroup>();
	chunkContextBindGroups = new Map<Chunk, GPUBindGroup>();
	chunkWorldPosBuffers = new Map<Chunk, GPUBuffer>();
	// Configuration uniforms
	configBuffer: GPUBuffer;
	// Timer for profiling
	timer: RenderTimer;
	// Per-chunk simulation state
	private chunkIterationCounts = new Map<Chunk, number>();
	private chunkNeedsUpdate = new Map<Chunk, boolean>();
	private maxIterations = 16; // Number of iterations to propagate light

	constructor() {
		this.timer = new RenderTimer("light");
		this.initBuffers();

		// Create compute pipeline
		this.pipeline = device.createComputePipeline({
			label: "Light Floodfill",
			layout: "auto",
			compute: {
				module: device.createShaderModule({
					code: shader,
				}),
				entryPoint: "main",
			},
		});

	}

	get renderTime(): number {
		return this.timer.renderTime;
	}

	update(encoder: GPUCommandEncoder, chunk: Chunk) {
		const needsUpdate = this.chunkNeedsUpdate.get(chunk) ?? false;
		if (!needsUpdate) return;

		const bindGroup = this.bindGroups.get(chunk);
		if (!bindGroup) return;

		// Start from the current lighting state
		encoder.copyBufferToBuffer(
			chunk.light,
			0,
			chunk.nextLight,
			0,
			chunk.light.size,
		);

		// Run multiple iterations of light propagation on the next buffer
		for (let i = 0; i < this.maxIterations; i++) {
			const pass = encoder.beginComputePass({
				label: `Light Propagation Iteration ${i}`,
				timestampWrites: i === 0 ? this.timer.getTimestampWrites() : undefined,
			});

			pass.setPipeline(this.pipeline);
			pass.setBindGroup(0, bindGroup);
			pass.setBindGroup(1, this.chunkContextBindGroups.get(chunk));

			// Dispatch with 4x4x4 workgroup size to match mesh generation
			const sSize = gridSize / compression;
			const workgroupsPerDim = Math.ceil(sSize / 4);
			pass.dispatchWorkgroups(
				workgroupsPerDim,
				workgroupsPerDim,
				workgroupsPerDim,
			);

			pass.end();
		}

		if (this.timer.getTimestampWrites()) {
			this.timer.resolveTimestamps(encoder);
		}

		// Swap buffers so the updated one becomes current
		this.swapBuffers(chunk);
		this.bindGroups.set(chunk, this.createComputeBindGroup(chunk));

		const iterationCount = (this.chunkIterationCounts.get(chunk) ?? 0) + 1;
		this.chunkIterationCounts.set(chunk, iterationCount);

		// Stop updating after the light has stabilized
		if (iterationCount >= this.maxIterations * 2) {
			this.chunkNeedsUpdate.set(chunk, false);
		}
	}

	// Force a light update (call when voxels are modified)
	invalidate(chunk?: Chunk) {
		if (chunk) {
			this.chunkNeedsUpdate.set(chunk, true);
			this.chunkIterationCounts.set(chunk, 0);
		} else {
			// Invalidate all chunks
			for (const c of this.chunkNeedsUpdate.keys()) {
				this.chunkNeedsUpdate.set(c, true);
				this.chunkIterationCounts.set(c, 0);
			}
		}
	}

	afterUpdate() {
		this.timer.readTimestamps();
	}

	registerChunk(chunk: Chunk) {
		// Initialize light buffer with skylight
		this.initializeLighting(chunk);

		this.bindGroups.set(chunk, this.createComputeBindGroup(chunk));

		// Mark chunk as needing light updates
		this.chunkNeedsUpdate.set(chunk, true);
		this.chunkIterationCounts.set(chunk, 0);

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

		// Create context bind group with chunk world position
		const contextBindGroup = device.createBindGroup({
			label: "Light Context",
			layout: this.pipeline.getBindGroupLayout(1),
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

		this.chunkContextBindGroups.set(chunk, contextBindGroup);
		this.chunkWorldPosBuffers.set(chunk, chunkWorldPosBuffer);
	}

	unregisterChunk(chunk: Chunk) {
		const worldPosBuffer = this.chunkWorldPosBuffers.get(chunk);
		if (worldPosBuffer) {
			worldPosBuffer.destroy();
		}
		this.bindGroups.delete(chunk);
		this.chunkContextBindGroups.delete(chunk);
		this.chunkWorldPosBuffers.delete(chunk);
		this.chunkNeedsUpdate.delete(chunk);
		this.chunkIterationCounts.delete(chunk);
	}

	private initBuffers() {


		// Configuration buffer for simulation parameters
		this.configBuffer = device.createBuffer({
			size: 32, // 8 * 4 bytes for configuration
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		// Initialize with default values
		this.updateConfig();
	}

	private updateConfig() {
		const config = new Float32Array(8);
		config[0] = this.maxIterations; // max_iterations
		config[1] = 0.85; // light_attenuation (how much light dims per step)
		config[2] = 0.95; // shadow_softness (how soft shadows are)
		config[3] = 1.0;  // skylight_intensity
		// config[4-7] reserved for future use

		device.queue.writeBuffer(this.configBuffer, 0, config);
	}

	private initializeLighting(chunk: Chunk) {
		// Initialize the light buffer with skylight from above
		const sSize = gridSize / compression;
		const totalCells = sSize * sSize * sSize;
		const initData = new Float32Array(totalCells * 2); // RG format

		// Calculate world Y position of this chunk
		const chunkWorldY = chunk.position[1] * gridSize;

		// Fill with initial skylight values - based on world position
		for (let z = 0; z < sSize; z++) {
			for (let y = 0; y < sSize; y++) {
				for (let x = 0; x < sSize; x++) {
					const index = (z * sSize * sSize + y * sSize + x) * 2;

					// Calculate world Y for this cell
					const worldY = chunkWorldY + y * compression;

					// Add skylight to top layer if in upper chunks (world Y > 0)
					const isTopLayer = (y >= sSize - 1);
					const isUpperChunk = worldY > 0;
					const skylight = (isTopLayer && isUpperChunk) ? Math.min(worldY / 256.0, 1.0) : 0.0;

					initData[index] = skylight;     // R: light intensity
					initData[index + 1] = isTopLayer ? 0.0 : 1.0;      // G: shadow factor
				}
			}
		}

		// Initialize both buffers with the same data
		device.queue.writeBuffer(chunk.light, 0, initData);
		device.queue.writeBuffer(chunk.nextLight, 0, initData);
	}

	private createComputeBindGroup(chunk: Chunk): GPUBindGroup {
		return device.createBindGroup({
			label: `Light Data Chunk ${chunk.id}`,
			layout: this.pipeline.getBindGroupLayout(0),
			entries: [
				{
					binding: 0,
					resource: {buffer: chunk.density},
				},
				{
					binding: 1,
					resource: {buffer: chunk.nextLight},
				},
				{
					binding: 2,
					resource: {buffer: this.configBuffer},
				},
			],
		});
	}

	private swapBuffers(chunk: Chunk) {
		const current = chunk.light;
		chunk.light = chunk.nextLight;
		chunk.nextLight = current;
	}
}
