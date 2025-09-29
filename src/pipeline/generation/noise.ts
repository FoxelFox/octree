import {contextUniform, device, gridSize} from '../../index';
import shader from './noise.wgsl' with {type: 'text'};
import {Chunk} from "../../chunk/chunk";

export class Noise {

	pipeline: GPUComputePipeline;
	bindGroup1: GPUBindGroup;

	data = new Map<Chunk, { bindGroup: GPUBindGroup, offsetBuffer: GPUBuffer }>();

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

		console.log('generate noise for chunk at', chunk.position);

		// Update chunk offset buffer with world-space position
		const chunkData = this.data.get(chunk);
		if (chunkData) {
			const offsetData = new Int32Array([
				chunk.position[0] * gridSize,
				chunk.position[1] * gridSize,
				chunk.position[2] * gridSize,
				0 // padding
			]);
			device.queue.writeBuffer(chunkData.offsetBuffer, 0, offsetData);
		}

		const computePass = commandEncoder.beginComputePass();
		computePass.setPipeline(this.pipeline);
		computePass.setBindGroup(0, chunkData?.bindGroup);
		computePass.setBindGroup(1, this.bindGroup1);
		computePass.dispatchWorkgroups(
			Math.ceil(gridSize / 4),
			Math.ceil(gridSize / 4),
			Math.ceil(gridSize / 4)
		);
		computePass.end();
	}

	registerChunk(chunk: Chunk) {

		// Create offset buffer for chunk world position
		const offsetBuffer = device.createBuffer({
			size: 16, // vec3<i32> + padding = 16 bytes
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		const bindGroup = device.createBindGroup({
			label: 'Noise',
			layout: this.pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: chunk.voxelData},
				{binding: 1, resource: {buffer: offsetBuffer}}
			],
		});

		this.data.set(chunk, { bindGroup, offsetBuffer });
	}

	unregisterChunk(chunk: Chunk) {
		const chunkData = this.data.get(chunk);
		if (chunkData) {
			chunkData.offsetBuffer.destroy();
		}
		this.data.delete(chunk);
	}
}
