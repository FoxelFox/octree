import {contextUniform, device, gridSize} from "../index";
import shader from "./distance_field.wgsl" with {type: "text"};

export class DistanceField {
	computePipeline!: GPUComputePipeline;
	bindGroup!: GPUBindGroup;
	distanceFieldBuffer!: GPUBuffer;
	contextBindGroup: GPUBindGroup;

	async init(noiseBuffer: GPUBuffer, contextUniformBuffer: GPUBuffer) {
		// Create distance field buffer (f32 distance + u32 material_id = 8 bytes per entry)
		const totalVoxels = gridSize * gridSize * gridSize;
		this.distanceFieldBuffer = device.createBuffer({
			size: totalVoxels * 8, // 4 bytes for distance + 4 bytes for material_id
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
		});

		// Load shader
		const shaderModule = device.createShaderModule({
			code: shader
		});

		// Create compute pipeline
		this.computePipeline = device.createComputePipeline({
			layout: 'auto',
			compute: {
				module: shaderModule,
				entryPoint: 'main',
			},
		});

		// Create bind group
		this.bindGroup = device.createBindGroup({
			layout: this.computePipeline.getBindGroupLayout(0),
			entries: [
				{
					binding: 0,
					resource: {buffer: noiseBuffer}, // Input: voxel data
				},
				{
					binding: 1,
					resource: {buffer: this.distanceFieldBuffer}, // Output: distance field
				},
			],
		});

		this.contextBindGroup = device.createBindGroup({
			layout: this.computePipeline.getBindGroupLayout(1),
			entries: [{
				binding: 0,
				resource: {buffer: contextUniform.uniformBuffer}
			}]
		});
	}

	update(encoder: GPUCommandEncoder) {
		const pass = encoder.beginComputePass();
		pass.setPipeline(this.computePipeline);
		pass.setBindGroup(0, this.bindGroup);
		pass.setBindGroup(1, this.contextBindGroup); // Context uniform

		// Dispatch with 4x4x4 workgroup size
		const workgroupsPerDim = Math.ceil(gridSize / 4);
		pass.dispatchWorkgroups(workgroupsPerDim, workgroupsPerDim, workgroupsPerDim);
		pass.end();
	}

	getDistanceFieldBuffer(): GPUBuffer {
		return this.distanceFieldBuffer;
	}
}