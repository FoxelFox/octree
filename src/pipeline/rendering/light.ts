import { compression, contextUniform, device, gridSize } from "../../index";
import { Mesh } from "../generation/mesh";
import shader from "./light.wgsl" with { type: "text" };
import { RenderTimer } from "../timing";

export class Light {
	// Double buffered light data buffers
	private lightBufferA: GPUBuffer;
	private lightBufferB: GPUBuffer;
	private currentBufferIndex = 0; // 0 = A, 1 = B
	
	// Compute pipeline for light propagation
	pipeline: GPUComputePipeline;
	bindGroup: GPUBindGroup;
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
	}

	private initBuffers() {
		const sSize = gridSize / compression;
		const totalCells = sSize * sSize * sSize;
		
		// Create both light buffers: stores light intensity (R) and shadow factor (G) for each compressed cell
		// Format: RG32Float - R = light intensity, G = shadow factor (0.0 = fully lit, 1.0 = fully shadowed)
		const bufferConfig = {
			size: totalCells * 8, // 2 * 4 bytes per cell (RG32Float)
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
		};
		this.lightBufferA = device.createBuffer(bufferConfig);
		this.lightBufferB = device.createBuffer(bufferConfig);

		// Configuration buffer for simulation parameters
		this.configBuffer = device.createBuffer({
			size: 32, // 8 * 4 bytes for configuration
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		// Initialize with default values
		this.updateConfig();
	}

	// Get the current buffer (for rendering/display)
	private getCurrentBuffer(): GPUBuffer {
		return this.currentBufferIndex === 0 ? this.lightBufferA : this.lightBufferB;
	}

	// Get the next buffer (for computing updates)
	private getNextBuffer(): GPUBuffer {
		return this.currentBufferIndex === 0 ? this.lightBufferB : this.lightBufferA;
	}

	// Swap buffers after update is complete
	private swapBuffers() {
		this.currentBufferIndex = 1 - this.currentBufferIndex;
	}

	// Create bind groups using the next buffer for computation
	private createBindGroups(mesh: Mesh) {
		this.bindGroup = device.createBindGroup({
			label: "Light Data",
			layout: this.pipeline.getBindGroupLayout(0),
			entries: [
				{
					binding: 0,
					resource: { buffer: mesh.density }, // Input: mesh density
				},
				{
					binding: 1,
					resource: { buffer: this.getNextBuffer() }, // Output: light data (next buffer)
				},
				{
					binding: 2,
					resource: { buffer: this.configBuffer }, // Config
				},
			],
		});
	}

	init(mesh: Mesh) {
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

		// Create bind groups - we'll recreate these when we swap buffers
		this.createBindGroups(mesh);

		this.contextBindGroup = device.createBindGroup({
			label: "Light Context",
			layout: this.pipeline.getBindGroupLayout(1),
			entries: [
				{
					binding: 0,
					resource: { buffer: contextUniform.uniformBuffer },
				},
			],
		});

		// Initialize light buffer with skylight
		this.initializeLighting();
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

	private initializeLighting() {
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
		device.queue.writeBuffer(this.lightBufferA, 0, initData);
		device.queue.writeBuffer(this.lightBufferB, 0, initData);
	}

	update(encoder: GPUCommandEncoder, mesh: Mesh) {
		if (!this.needsUpdate) return;

		// First, copy current buffer to next buffer to preserve existing lighting
		encoder.copyBufferToBuffer(
			this.getCurrentBuffer(), 0,
			this.getNextBuffer(), 0,
			this.getCurrentBuffer().size
		);

		// Run multiple iterations of light propagation on the next buffer
		for (let i = 0; i < this.maxIterations; i++) {
			const pass = encoder.beginComputePass({
				label: `Light Propagation Iteration ${i}`,
				timestampWrites: i === 0 ? this.timer.getTimestampWrites() : undefined,
			});

			pass.setPipeline(this.pipeline);
			pass.setBindGroup(0, this.bindGroup);
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
		this.swapBuffers();
		
		// Recreate bind groups for the new configuration
		this.createBindGroups(mesh);

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

	// Get the light buffer for use in other shaders
	getLightBuffer(): GPUBuffer {
		return this.getCurrentBuffer();
	}

	afterUpdate() {
		this.timer.readTimestamps();
	}

	get renderTime(): number {
		return this.timer.renderTime;
	}

	// Configuration methods
	setMaxIterations(iterations: number) {
		this.maxIterations = Math.max(1, Math.min(32, iterations));
		this.updateConfig();
	}

	setLightAttenuation(attenuation: number) {
		const config = new Float32Array([
			this.maxIterations,
			Math.max(0.1, Math.min(1.0, attenuation)),
			0.95, // shadow_softness
			1.0,  // skylight_intensity
		]);
		device.queue.writeBuffer(this.configBuffer, 0, config);
	}
}