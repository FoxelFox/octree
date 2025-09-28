import {contextUniform, device, gridSize} from '../../index';
import shader from './noise.wgsl' with {type: 'text'};
import {Chunk} from "../../chunk/chunk";

export class Noise {

	pipeline: GPUComputePipeline;
	bindGroup1: GPUBindGroup;

	data = new Map<Chunk, GPUBindGroup>();

	constructor() {

		this.pipeline = device.createComputePipeline({
			label: 'Noise',
			layout: 'auto',
			compute: {
				module: device.createShaderModule({
					code: shader,
				}),
				entryPoint: 'main',
			},
		});


		this.bindGroup1 = device.createBindGroup({
			layout: this.pipeline.getBindGroupLayout(1),
			entries: [
				{
					binding: 0,
					resource: {buffer: contextUniform.uniformBuffer},
				},
			],
		});
	}

	update(commandEncoder: GPUCommandEncoder, chunk: Chunk) {

		console.log('generate noise');
		const computePass = commandEncoder.beginComputePass();
		computePass.setPipeline(this.pipeline);
		computePass.setBindGroup(0, this.data.get(chunk));
		computePass.setBindGroup(1, this.bindGroup1);
		computePass.dispatchWorkgroups(
			Math.ceil(gridSize / 4),
			Math.ceil(gridSize / 4),
			Math.ceil(gridSize / 4)
		);
		computePass.end();
	}

	registerChunk(chunk: Chunk) {

		const bindGroup = device.createBindGroup({
			label: 'Noise',
			layout: this.pipeline.getBindGroupLayout(0),
			entries: [{binding: 0, resource: chunk.voxelData}],
		});

		this.data.set(chunk, bindGroup);
	}

	unregisterChunk(chunk: Chunk) {
		this.data.delete(chunk);
	}
}
