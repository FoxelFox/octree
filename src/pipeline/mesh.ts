import {contextUniform, device, gridSize} from "../index";
import shader from "./mesh.wgsl";
import {Noise} from "./noise";

export class Mesh {
	pipeline: GPUComputePipeline;
	bindGroup: GPUBindGroup;
	contextBindGroup: GPUBindGroup;
	meshes: GPUBuffer;
	commands: GPUBuffer;
	density: GPUBuffer;

	init(noise: Noise) {
		const sSize = gridSize / 8;
		const maxMeshCount = sSize * sSize * sSize;
		const maxMeshSize =
			4           // vertexCount (u32 = 4 bytes)
			+ 12        // padding to align vertices array to 16-byte boundary
			+ 1280 * 16 // vertices (vec4<f32> = 16 bytes each)
			+ 1280 * 16 // normals (vec3<f32> = 16 bytes each in array, padded)

		this.meshes = device.createBuffer({
			size: maxMeshSize * maxMeshCount,
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
			usage:
				GPUBufferUsage.STORAGE |
				GPUBufferUsage.COPY_SRC
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
					resource: {buffer: noise.noiseBuffer}, // Input
				},
				{
					binding: 1,
					resource: {buffer: this.meshes}, // Output
				},
				{
					binding: 2,
					resource: {buffer: this.commands}, // Output
				},
				{
					binding: 3,
					resource: {buffer: this.density}, // Output
				},
			],
		});

		this.contextBindGroup = device.createBindGroup({
			label: "Mesh Context",
			layout: this.pipeline.getBindGroupLayout(1),
			entries: [
				{
					binding: 0,
					resource: {buffer: contextUniform.uniformBuffer},
				},
			],
		});
	}

	update(encoder: GPUCommandEncoder) {
		const pass = encoder.beginComputePass();
		pass.setPipeline(this.pipeline);
		pass.setBindGroup(0, this.bindGroup);
		pass.setBindGroup(1, this.contextBindGroup);

		// Dispatch with 4x4x4 workgroup size
		const workgroupsPerDim = Math.ceil(gridSize / 8 / 4);
		pass.dispatchWorkgroups(
			workgroupsPerDim,
			workgroupsPerDim,
			workgroupsPerDim,
		);
		pass.end();
	}
}
