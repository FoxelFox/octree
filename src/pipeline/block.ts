import {context, contextUniform, device, gridSize} from "../index";
import shader from "./block.wgsl" with {type: "text"};
import {Mesh} from "./mesh";

export class Block {

	// input
	mesh: Mesh;

	pipeline: GPURenderPipeline
	bindGroup: GPUBindGroup
	uniformBindGroup: GPUBindGroup
	initialized: boolean;


	constructor() {

	}

	update(commandEncoder: GPUCommandEncoder) {

		if (!this.initialized) {
			this.init();
		}

		const pass = commandEncoder.beginRenderPass({
			label: 'Block',
			colorAttachments: [{
				view: context.getCurrentTexture().createView(),
				loadOp: 'clear',
				storeOp: 'store',
				clearValue: {r: 0, g: 0, b: 0, a: 0}
			}]
		});

		pass.setPipeline(this.pipeline);
		pass.setBindGroup(0, this.bindGroup);
		pass.setBindGroup(1, this.uniformBindGroup);
		
		// Draw instances for each mesh chunk
		const sSize = gridSize / 8;
		const maxInstances = sSize * sSize * sSize;
		pass.draw(1024 * 6, maxInstances); // Max vertices per mesh * max instances
		pass.end();

	}

	afterUpdate() {

	}

	init() {

		this.pipeline = device.createRenderPipeline({
			label: 'Block Indices',
			layout: "auto",
			fragment: {
				module: device.createShaderModule({
					label: 'Block Fragment Shader',
					code: shader,
				}),
				targets: [
					{format: 'bgra8unorm'}
				]
			},
			vertex: {
				module: device.createShaderModule({
					label: 'Block Vertex Shader',
					code: shader
				})
			},
			primitive: {
				topology: 'triangle-list'
			}
		});

		this.bindGroup = device.createBindGroup({
			label: 'Block',
			layout: this.pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: this.mesh.meshes}
			]
		});

		this.uniformBindGroup = device.createBindGroup({
			label: 'Block BindGroup for Context',
			layout: this.pipeline.getBindGroupLayout(1),
			entries: [
				{binding: 0, resource: contextUniform.uniformBuffer}
			]
		});

		this.initialized = true;
	}

	to3D(index: number) {
		const size = gridSize;
		const z = Math.floor(index / (size * size));
		const y = Math.floor((index % (size * size)) / size);
		const x = index % size;
		return [x, y, z];
	}

	get vertices(): Float32Array {
		return new Float32Array([
			1, 0, 1, 1,
			0, 0, 1, 1,
			0, 0, 0, 1,
			1, 0, 0, 1,
			1, 0, 1, 1,
			0, 0, 0, 1,

			1, 1, 1, 1,
			1, 0, 1, 1,
			1, 0, 0, 1,
			1, 1, 0, 1,
			1, 1, 1, 1,
			1, 0, 0, 1,

			0, 1, 1, 1,
			1, 1, 1, 1,
			1, 1, 0, 1,
			0, 1, 0, 1,
			0, 1, 1, 1,
			1, 1, 0, 1,

			0, 0, 1, 1,
			0, 1, 1, 1,
			0, 1, 0, 1,
			0, 0, 0, 1,
			0, 0, 1, 1,
			0, 1, 0, 1,

			1, 1, 1, 1,
			0, 1, 1, 1,
			0, 0, 1, 1,
			0, 0, 1, 1,
			1, 0, 1, 1,
			1, 1, 1, 1,

			1, 0, 0, 1,
			0, 0, 0, 1,
			0, 1, 0, 1,
			1, 1, 0, 1,
			1, 0, 0, 1,
			0, 1, 0, 1
		]);
	}
}