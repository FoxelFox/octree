import { contextUniform, device, gridSize } from "..";
import { Mesh } from "./mesh";
import { Noise } from "./noise";
import shader from "./cull.wgsl" with { type: "text" };

export class Cull {
	counter: GPUBuffer;
	bindGroup: GPUBindGroup;
	contextBindGroup: GPUBindGroup;
	pipeline: GPUComputePipeline;
	indices: GPUBuffer;

	init(noise: Noise, mesh: Mesh) {
		this.counter = device.createBuffer({
			size: 4,
			usage:
				GPUBufferUsage.STORAGE |
				GPUBufferUsage.COPY_SRC |
				GPUBufferUsage.COPY_DST,
		});

		this.indices = device.createBuffer({
			size: Math.pow(gridSize / 8, 3),
			usage:
				GPUBufferUsage.STORAGE |
				GPUBufferUsage.COPY_SRC |
				GPUBufferUsage.COPY_DST,
		});

		device.queue.writeBuffer(this.counter, 0, new Uint32Array([0]));

		this.pipeline = device.createComputePipeline({
			layout: "auto",
			label: "Cull",
			compute: {
				module: device.createShaderModule({
					code: shader,
				}),
				entryPoint: "main",
			},
		});

		this.bindGroup = device.createBindGroup({
			layout: this.pipeline.getBindGroupLayout(0),
			label: "Cull",
			entries: [
				{
					binding: 0,
					resource: this.counter,
				},
				{
					binding: 1,
					resource: mesh.meshes,
				},
				{
					binding: 2,
					resource: this.indices,
				},
			],
		});

		this.contextBindGroup = device.createBindGroup({
			layout: this.pipeline.getBindGroupLayout(1),
			entries: [
				{
					binding: 0,
					resource: { buffer: contextUniform.uniformBuffer },
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

	async readback(): Promise<void> {
		const readBuffer = device.createBuffer({
			size: this.indices.size,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
		});

		const encoder = device.createCommandEncoder();
		encoder.copyBufferToBuffer(
			this.indices,
			0,
			readBuffer,
			0,
			this.indices.size,
		);
		device.queue.submit([encoder.finish()]);

		await readBuffer.mapAsync(GPUMapMode.READ);
		const data = readBuffer.getMappedRange();
		const result = data.slice();
		readBuffer.unmap();
		readBuffer.destroy();

		console.log(new Uint32Array(result));
	}
}
