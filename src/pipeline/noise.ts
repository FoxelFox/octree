import {contextUniform, device} from "..";
import shader from "./noise.wgsl" with {type: "text"};

export class Noise {

	noiseBuffer: GPUBuffer
	maxDepth: number = 2
	uniformBuffer: GPUBuffer
	pipeline: GPUComputePipeline
	bindGroup0: GPUBindGroup
	bindGroup1: GPUBindGroup
	readbackBuffer: GPUBuffer

	constructor() {
		const size = Math.pow(this.gridSize, 3) * 4;
		this.noiseBuffer = device.createBuffer({
			label: "Noise",
			size,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
		});

		this.uniformBuffer = device.createBuffer({
			label: "Noise Uniform",
			size: 4,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});

		this.readbackBuffer = device.createBuffer({
			size,
			usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
		});

		device.queue.writeBuffer(this.uniformBuffer, 0, new Uint32Array([this.gridSize]));

		this.pipeline = device.createComputePipeline({
			label: "Noise",
			layout: "auto",
			compute: {
				module: device.createShaderModule({
					code: shader,
				}),
				entryPoint: "main",
			},
		});

		this.bindGroup0 = device.createBindGroup({
			label: "Noise",
			layout: this.pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: this.uniformBuffer},
				{binding: 1, resource: this.noiseBuffer},
			]
		});

		this.bindGroup1 = device.createBindGroup({
			layout: this.pipeline.getBindGroupLayout(1),
			entries: [{
				binding: 0,
				resource: {buffer: contextUniform.uniformBuffer}
			}]
		});
	}

	update(commandEncoder: GPUCommandEncoder) {

		const computePass = commandEncoder.beginComputePass();
		computePass.setPipeline(this.pipeline);
		computePass.setBindGroup(0, this.bindGroup0);
		computePass.setBindGroup(1, this.bindGroup1);
		console.log(this.gridSize)
		computePass.dispatchWorkgroups(
			Math.ceil(this.gridSize / 4),
			Math.ceil(this.gridSize / 4),
			Math.ceil(this.gridSize / 4)
		);
		computePass.end();

		const size = Math.pow(this.gridSize, 3) * 4;
		commandEncoder.copyBufferToBuffer(this.noiseBuffer, 0, this.readbackBuffer, 0, size);

	}

	afterUpdate(commandEncoder: GPUCommandEncoder) {
		this.readbackBuffer.mapAsync(GPUMapMode.READ).then(() => {
			const result = new Uint32Array(this.readbackBuffer.getMappedRange());
			console.log("Noise:", result)
			this.readbackBuffer.unmap();
		});
	}

	get gridSize(): number {
		return Math.pow(2, this.maxDepth);
	}
}