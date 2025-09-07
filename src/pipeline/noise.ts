import {contextUniform, device, gridSize} from "..";
import shader from "./noise.wgsl" with {type: "text"};

export class Noise {

	noiseBuffer: GPUBuffer
	pipeline: GPUComputePipeline
	bindGroup0: GPUBindGroup
	bindGroup1: GPUBindGroup


	// output
	result: Float32Array;

	constructor() {
		const size = Math.pow(gridSize, 3) * 4;
		this.noiseBuffer = device.createBuffer({
			label: "Noise",
			size,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
		});

		this.pipeline = device.createComputePipeline({
			label: "Noise",
			layout: "auto",
			compute: {
				module: device.createShaderModule({
					code: shader
				}),
				entryPoint: "main",
			},
		});

		this.bindGroup0 = device.createBindGroup({
			label: "Noise",
			layout: this.pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: this.noiseBuffer},
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

		if (this.result) {
			return;
		}

		console.log('generate noise')
		const computePass = commandEncoder.beginComputePass();
		computePass.setPipeline(this.pipeline);
		computePass.setBindGroup(0, this.bindGroup0);
		computePass.setBindGroup(1, this.bindGroup1);
		computePass.dispatchWorkgroups(
			Math.ceil(gridSize / 4),
			Math.ceil(gridSize / 4),
			Math.ceil(gridSize / 4)
		);
		computePass.end();
	}
}