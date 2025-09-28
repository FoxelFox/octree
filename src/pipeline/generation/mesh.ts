import {compression, contextUniform, device, gridSize} from '../../index';
import shader from './mesh.wgsl';
import {RenderTimer} from '../timing';
import {EDGE_TABLE, TRIANGLE_TABLE} from './marchingCubeTables';
import {Chunk} from "../../chunk/chunk";

export class Mesh {
	pipeline: GPUComputePipeline;
	contextBindGroup: GPUBindGroup;
	offsetBuffer: GPUBuffer;
	edgeTableBuffer: GPUBuffer;
	triangleTableBuffer: GPUBuffer;
	timer: RenderTimer;

	chunkBindGroups = new Map<Chunk, GPUBindGroup>();

	constructor() {
		this.timer = new RenderTimer('mesh');

		this.offsetBuffer = device.createBuffer({
			size: 16, // vec3<u32> + padding = 16 bytes
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		// Create and initialize lookup table buffers
		this.edgeTableBuffer = device.createBuffer({
			size: EDGE_TABLE.byteLength,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});
		device.queue.writeBuffer(this.edgeTableBuffer, 0, EDGE_TABLE);

		this.triangleTableBuffer = device.createBuffer({
			size: TRIANGLE_TABLE.byteLength,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});
		device.queue.writeBuffer(this.triangleTableBuffer, 0, TRIANGLE_TABLE);

		const shaderModule = device.createShaderModule({
			code: shader,
		});

		this.pipeline = device.createComputePipeline({
			label: 'Mesh',
			layout: 'auto',
			compute: {
				module: shaderModule,
				entryPoint: 'main',
			},
		});


		this.contextBindGroup = device.createBindGroup({
			label: 'Mesh Context',
			layout: this.pipeline.getBindGroupLayout(1),
			entries: [
				{
					binding: 0,
					resource: {buffer: contextUniform.uniformBuffer},
				},
				{
					binding: 1,
					resource: {buffer: this.offsetBuffer},
				},
				{
					binding: 2,
					resource: {buffer: this.edgeTableBuffer},
				},
				{
					binding: 3,
					resource: {buffer: this.triangleTableBuffer},
				},
			],
		});
	}

	get renderTime(): number {
		return this.timer.renderTime;
	}

	update(
		encoder: GPUCommandEncoder,
		chunk: Chunk,
		bounds?: { min: [number, number, number]; max: [number, number, number] }
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
			pass.setBindGroup(0, this.chunkBindGroups.get(chunk));
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
			pass.setBindGroup(0, this.chunkBindGroups.get(chunk));
			pass.setBindGroup(1, this.contextBindGroup);

			// Dispatch with 4x4x4 workgroup size for full grid
			const workgroupsPerDim = Math.ceil(gridSize / compression / 4);
			pass.dispatchWorkgroups(
				workgroupsPerDim,
				workgroupsPerDim,
				workgroupsPerDim
			);
			pass.end();
		}

		this.timer.resolveTimestamps(encoder);
	}

	afterUpdate() {
		this.timer.readTimestamps();
	}

	registerChunk(chunk: Chunk) {

		const bindGroup = device.createBindGroup({
			label: 'Mesh',
			layout: this.pipeline.getBindGroupLayout(0),
			entries: [
				{
					binding: 0,
					resource: {buffer: chunk.voxelData}, // Input
				},
				{
					binding: 1,
					resource: {buffer: chunk.vertexCounts}, // Output
				},
				{
					binding: 2,
					resource: {buffer: chunk.vertices}, // Output
				},
				{
					binding: 3,
					resource: {buffer: chunk.normals}, // Output
				},
				{
					binding: 4,
					resource: {buffer: chunk.colors}, // Output
				},
				{
					binding: 5,
					resource: {buffer: chunk.commands}, // Output
				},
				{
					binding: 6,
					resource: {buffer: chunk.density}, // Output
				},
			],
		});

		this.chunkBindGroups.set(chunk, bindGroup);
	}

	unregisterChunk(chunk: Chunk) {
		this.chunkBindGroups.delete(chunk);
	}
}
