import {contextUniform, device, gridSize} from '../../index';
import shader from './noise.wgsl' with {type: 'text'};
import {Chunk} from "../../chunk/chunk";

export class Noise {

	pipeline: GPUComputePipeline;
	bindGroup1: GPUBindGroup;

	data = new Map<Chunk, { bindGroup: GPUBindGroup, offsetBuffer: GPUBuffer, paramsBuffer: GPUBuffer }>();

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

	update(
		commandEncoder: GPUCommandEncoder,
		chunk: Chunk,
		sliceOffset = 0,
		sliceCount?: number,
	) {
		const chunkData = this.data.get(chunk);
		if (!chunkData) {
			return;
		}

		// Update chunk offset buffer with world-space position
		const offsetData = new Int32Array([
			chunk.position[0] * gridSize,
			chunk.position[1] * gridSize,
			chunk.position[2] * gridSize,
			0 // padding
		]);
		device.queue.writeBuffer(chunkData.offsetBuffer, 0, offsetData);

		const voxelGridSize = gridSize + 1;
		const clampedOffset = Math.min(Math.max(sliceOffset, 0), voxelGridSize);
		const effectiveCount = sliceCount ?? voxelGridSize;
		const clampedCount = Math.min(effectiveCount, voxelGridSize - clampedOffset);
		if (clampedCount <= 0) {
			return;
		}

		const paramsData = new Uint32Array([
			clampedOffset,
			clampedCount,
			0,
			0,
		]);
		device.queue.writeBuffer(chunkData.paramsBuffer, 0, paramsData);

		const computePass = commandEncoder.beginComputePass();
		computePass.setPipeline(this.pipeline);
		computePass.setBindGroup(0, chunkData.bindGroup);
		computePass.setBindGroup(1, this.bindGroup1);

		const workgroupsPerDim = Math.ceil(voxelGridSize / 4);
		const workgroupsZ = Math.max(1, Math.ceil(clampedCount / 4));
		computePass.dispatchWorkgroups(
			workgroupsPerDim,
			workgroupsPerDim,
			workgroupsZ,
		);
		computePass.end();
	}

	registerChunk(chunk: Chunk) {

		// Create offset buffer for chunk world position
		const chunkLabel = `Chunk[${chunk.id}](${chunk.position[0]},${chunk.position[1]},${chunk.position[2]})`;
		const offsetBuffer = device.createBuffer({
			label: `${chunkLabel} Noise Offset`,
			size: 16, // vec3<i32> + padding = 16 bytes
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		const paramsBuffer = device.createBuffer({
			label: `${chunkLabel} Noise Slice Params`,
			size: 16,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		const bindGroup = device.createBindGroup({
			label: 'Noise',
			layout: this.pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: chunk.voxelData},
				{binding: 1, resource: {buffer: offsetBuffer}},
				{binding: 2, resource: {buffer: paramsBuffer}},
			],
		});

		this.data.set(chunk, {bindGroup, offsetBuffer, paramsBuffer});
}

	unregisterChunk(chunk: Chunk) {
		const chunkData = this.data.get(chunk);
		if (chunkData) {
			chunkData.offsetBuffer.destroy();
			chunkData.paramsBuffer.destroy();
		}
		this.data.delete(chunk);
	}
}
