import {contextUniform, device, gridSize} from "../index";
import shader from "./mesh.wgsl";
import {Noise} from "./noise";

export class Mesh {

	pipeline: GPUComputePipeline
	bindGroup: GPUBindGroup
	contextBindGroup: GPUBindGroup
	meshes: GPUBuffer

	init(noise: Noise) {
		const sSize = gridSize / 8;
		const maxMeshCount = sSize * sSize * sSize
		const maxMeshSize = 1024 * 16 + 4

		this.meshes = device.createBuffer({
			size: maxMeshSize * maxMeshCount,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
		});


		const shaderModule = device.createShaderModule({
			code: shader
		});

		this.pipeline = device.createComputePipeline({
			layout: 'auto',
			compute: {
				module: shaderModule,
				entryPoint: 'main',
			},
		});

		this.bindGroup = device.createBindGroup({
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
			],
		});

		this.contextBindGroup = device.createBindGroup({
			layout: this.pipeline.getBindGroupLayout(1),
			entries: [{
				binding: 0,
				resource: {buffer: contextUniform.uniformBuffer}
			}]
		});
	}

	update(encoder: GPUCommandEncoder) {
		const pass = encoder.beginComputePass();
		pass.setPipeline(this.pipeline);
		pass.setBindGroup(0, this.bindGroup);
		pass.setBindGroup(1, this.contextBindGroup);

		// Dispatch with 4x4x4 workgroup size
		const workgroupsPerDim = Math.ceil(gridSize / 8 / 4);
		pass.dispatchWorkgroups(workgroupsPerDim, workgroupsPerDim, workgroupsPerDim);
		pass.end();

	}

	async readback(): Promise<void> {
		const readBuffer = device.createBuffer({
			size: this.meshes.size,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
		});

		const encoder = device.createCommandEncoder();
		encoder.copyBufferToBuffer(this.meshes, 0, readBuffer, 0, this.meshes.size);
		device.queue.submit([encoder.finish()]);

		await readBuffer.mapAsync(GPUMapMode.READ);
		const data = readBuffer.getMappedRange();
		const result = data.slice();
		readBuffer.unmap();
		readBuffer.destroy();

		console.log(new Float32Array(result));
	}

}