import { compression, contextUniform, device, gridSize } from "../../index";
import shader from "./mesh.wgsl";
import { Noise } from "./noise";
import { RenderTimer } from "../timing";

export class Mesh {
	pipeline: GPUComputePipeline;
	bindGroup: GPUBindGroup;
	contextBindGroup: GPUBindGroup;
	vertexCounts: GPUBuffer;
	vertices: GPUBuffer;
	normals: GPUBuffer;
	colors: GPUBuffer;
	commands: GPUBuffer;
	density: GPUBuffer;
	offsetBuffer: GPUBuffer;
	timer: RenderTimer;

	init(noise: Noise) {
		this.timer = new RenderTimer("mesh");

		const sSize = gridSize / compression;
		const maxMeshCount = sSize * sSize * sSize;
		const maxVertices = maxMeshCount * 1536; // Maximum vertices across all meshes

		// Separate buffers for mesh data
		this.vertexCounts = device.createBuffer({
			size: 4 * maxMeshCount, // u32 = 4 bytes each
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
		});

		this.vertices = device.createBuffer({
			size: 8 * maxVertices, // vec4<f16> = 8 bytes each
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
		});

		this.normals = device.createBuffer({
			size: 8 * maxVertices, // vec3<f16> = 8 bytes each (with padding)
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
		});

		this.colors = device.createBuffer({
			size: 4 * maxVertices, // u32 = 4 bytes each
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
		});

		this.commands = device.createBuffer({
			size: 16 * maxMeshCount,
			usage:
				GPUBufferUsage.STORAGE |
				GPUBufferUsage.COPY_SRC |
				GPUBufferUsage.INDIRECT,
		});

		this.density = device.createBuffer({
			size: 4 * maxMeshCount,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
		});

		this.offsetBuffer = device.createBuffer({
			size: 16, // vec3<u32> + padding = 16 bytes
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		const shaderModule = device.createShaderModule({
			code: shader,
		});

		this.pipeline = device.createComputePipeline({
			label: "Mesh",
			layout: "auto",
			compute: {
				module: shaderModule,
				entryPoint: "main",
			},
		});

		this.bindGroup = device.createBindGroup({
			label: "Mesh",
			layout: this.pipeline.getBindGroupLayout(0),
			entries: [
				{
					binding: 0,
					resource: { buffer: noise.noiseBuffer }, // Input
				},
				{
					binding: 1,
					resource: { buffer: this.vertexCounts }, // Output
				},
				{
					binding: 2,
					resource: { buffer: this.vertices }, // Output
				},
				{
					binding: 3,
					resource: { buffer: this.normals }, // Output
				},
				{
					binding: 4,
					resource: { buffer: this.colors }, // Output
				},
				{
					binding: 5,
					resource: { buffer: this.commands }, // Output
				},
				{
					binding: 6,
					resource: { buffer: this.density }, // Output
				},
			],
		});

		this.contextBindGroup = device.createBindGroup({
			label: "Mesh Context",
			layout: this.pipeline.getBindGroupLayout(1),
			entries: [
				{
					binding: 0,
					resource: { buffer: contextUniform.uniformBuffer },
				},
				{
					binding: 1,
					resource: { buffer: this.offsetBuffer },
				},
			],
		});
	}

	update(
		encoder: GPUCommandEncoder,
		bounds?: { min: [number, number, number]; max: [number, number, number] },
	) {
		if (bounds) {
			// Calculate mesh chunk bounds in compressed grid space
			const sSize = gridSize / compression;
			const minChunk = [
				Math.max(0, Math.floor(bounds.min[0] / compression)),
				Math.max(0, Math.floor(bounds.min[1] / compression)),
				Math.max(0, Math.floor(bounds.min[2] / compression)),
			];
			const maxChunk = [
				Math.min(sSize - 1, Math.ceil(bounds.max[0] / compression)),
				Math.min(sSize - 1, Math.ceil(bounds.max[1] / compression)),
				Math.min(sSize - 1, Math.ceil(bounds.max[2] / compression)),
			];

			// Write offset to shader uniform
			const offsetData = new Uint32Array([
				minChunk[0],
				minChunk[1],
				minChunk[2],
				0,
			]); // padding for vec3<u32>
			device.queue.writeBuffer(this.offsetBuffer, 0, offsetData);

			const pass = encoder.beginComputePass({
				timestampWrites: this.timer.getTimestampWrites(),
			});
			pass.setPipeline(this.pipeline);
			pass.setBindGroup(0, this.bindGroup);
			pass.setBindGroup(1, this.contextBindGroup);

			// Dispatch only the affected chunks with 4x4x4 workgroup size
			const workgroupsX = Math.ceil((maxChunk[0] - minChunk[0] + 1) / 4);
			const workgroupsY = Math.ceil((maxChunk[1] - minChunk[1] + 1) / 4);
			const workgroupsZ = Math.ceil((maxChunk[2] - minChunk[2] + 1) / 4);

			pass.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);
			pass.end();
		} else {
			// Write zero offset for full grid update
			const offsetData = new Uint32Array([0, 0, 0, 0]);
			device.queue.writeBuffer(this.offsetBuffer, 0, offsetData);

			const pass = encoder.beginComputePass({
				timestampWrites: this.timer.getTimestampWrites(),
			});
			pass.setPipeline(this.pipeline);
			pass.setBindGroup(0, this.bindGroup);
			pass.setBindGroup(1, this.contextBindGroup);

			// Dispatch with 4x4x4 workgroup size for full grid
			const workgroupsPerDim = Math.ceil(gridSize / compression / 4);
			pass.dispatchWorkgroups(
				workgroupsPerDim,
				workgroupsPerDim,
				workgroupsPerDim,
			);
			pass.end();
		}

		this.timer.resolveTimestamps(encoder);
	}

	afterUpdate() {
		this.timer.readTimestamps();
	}

	get renderTime(): number {
		return this.timer.renderTime;
	}
}
