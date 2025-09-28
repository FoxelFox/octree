import {compression, contextUniform, device, gridSize} from "../../index";
import shader from "./light.wgsl" with {type: "text"};
import {RenderTimer} from "../timing";
import {Chunk} from "../../chunk/chunk";

export class Light {

	// Compute pipeline for light propagation
	pipeline: GPUComputePipeline;
	bindGroups = new Map<Chunk, GPUBindGroup>();
	contextBindGroup: GPUBindGroup;
	// Configuration uniforms
	configBuffer: GPUBuffer;
	// Timer for profiling
	timer: RenderTimer;
	// Simulation state
	private iterationCount = 0;
	private maxIterations = 16; // Number of iterations to propagate light
	private needsUpdate = true;

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

		this.contextBindGroup = device.createBindGroup({
			label: "Light Context",
			layout: this.pipeline.getBindGroupLayout(1),
			entries: [
				{
					binding: 0,
					resource: {buffer: contextUniform.uniformBuffer},
				},
			],
		});
	}

	get renderTime(): number {
		return this.timer.renderTime;
	}

	update(encoder: GPUCommandEncoder, chunk: Chunk) {
		if (!this.needsUpdate) return;

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
			pass.setBindGroup(1, this.contextBindGroup);

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

		this.iterationCount++;
		// Stop updating after the light has stabilized
		if (this.iterationCount >= this.maxIterations * 2) {
			this.needsUpdate = false;
		}
	}

	// Force a light update (call when voxels are modified)
	invalidate() {
		this.needsUpdate = true;
		this.iterationCount = 0;
		// No need to reset lighting - the double buffering will naturally
		// recalculate lighting from the current state without flicker
	}

	afterUpdate() {
		this.timer.readTimestamps();
	}

	registerChunk(chunk: Chunk) {
		// Initialize light buffer with skylight
		this.initializeLighting(chunk);

		this.bindGroups.set(chunk, this.createComputeBindGroup(chunk));
	}

	unregisterChunk(chunk: Chunk) {
		this.bindGroups.delete(chunk);
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

		// Fill with initial skylight values - but be more conservative
		for (let z = 0; z < sSize; z++) {
			for (let y = 0; y < sSize; y++) {
				for (let x = 0; x < sSize; x++) {
					const index = (z * sSize * sSize + y * sSize + x) * 2;

					// Only add skylight to the very top layer, everything else starts dark
					const skylight = (y >= sSize - 1) ? 1.0 : 0.0;

					initData[index] = skylight;     // R: light intensity
					initData[index + 1] = (y >= sSize - 1) ? 0.0 : 1.0;      // G: shadow factor
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
