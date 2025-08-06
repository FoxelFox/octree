import {Noise} from "./noise";
import {context, contextUniform, device} from "../index";
import shader from "./block.wgsl" with {type: "text"};

export class Block {

	// input
	noise: Noise;

	pipeline: GPURenderPipeline
	bindGroup: GPUBindGroup
	uniformBindGroup: GPUBindGroup
	indexBuffer: GPUBuffer
	vertexBuffer: GPUBuffer


	constructor() {

	}

	update(commandEncoder: GPUCommandEncoder) {

		if (!this.noise.result) {

			console.log('no voxel currently')
			return;
		}

		if (this.indexBuffer) {
			this.indexBuffer.destroy();
		}

		const positions = []
		for (let i = 0; i < this.noise.result.length; i++) {
			if(this.noise.result[i] === 1) {
				const [x,y,z] = this.to3D(i);
				positions.push(...[x,y,z,0]);
			}
		}

		this.indexBuffer = device.createBuffer({
			label: 'Block Positions',
			size: positions.length * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});

		const vertices = this.vertices;
		this.vertexBuffer = device.createBuffer({
			label: 'Block Vertices',
			size: vertices.byteLength,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		})

		device.queue.writeBuffer(this.indexBuffer, 0, new Float32Array(positions));
		device.queue.writeBuffer(this.vertexBuffer, 0, vertices);

		this.pipeline = device.createRenderPipeline({
			label: 'Block Indices',
			layout: "auto",
			fragment: {
				module: device.createShaderModule({
					label: 'Block Fragment Shader',
					code: shader,
				}),
				targets: [
					{format: 'bgra8unorm'},
					{format: 'bgra8unorm'}
				]
			},
			vertex: {
				module: device.createShaderModule({
					label: 'Block Vertex Shader',
					code: shader
				}),
				buffers: [{
					arrayStride: 4 * 4,
					attributes: [{
						// vertex position
						shaderLocation: 0,
						format: "float32x4",
						offset: 0
					}]
				}]
			},
			primitive: {
				topology: 'triangle-list'
			}
		});

		this.bindGroup = device.createBindGroup({
			label: 'Block',
			layout: this.pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: this.indexBuffer}
			]
		});

		this.uniformBindGroup = device.createBindGroup({
			label: 'Block BindGroup for Context',
			layout: this.pipeline.getBindGroupLayout(1),
			entries: [
				{binding: 0, resource: contextUniform.uniformBuffer}
			]
		})

		const pass = commandEncoder.beginRenderPass({
			label: 'Block',
			colorAttachments: [{
				view: context.getCurrentTexture().createView(),
				loadOp: 'clear',
				storeOp: 'store',
				clearValue: {r: 0, g: 0, b: 0, a: 1}
			}]
		});

		pass.setPipeline(this.pipeline);
		pass.setBindGroup(0, this.bindGroup);
		pass.setBindGroup(1, this.uniformBindGroup);
		pass.setVertexBuffer(0, this.vertexBuffer);
		pass.draw(this.vertexBuffer.size / 16, this.indexBuffer.size / 4);
		pass.end();

	}

	afterUpdate() {

	}

	to3D(index: number) {
		const size = this.noise.gridSize;
		const z = Math.floor(index / (size * size));
		const y = Math.floor((index % (size * size)) / size);
		const x = index % size;
		return [x, y, z];
	}

	get vertices(): Float32Array {
		return new Float32Array([
			1, -1, 1, 1,
			-1, -1, 1, 1,
			-1, -1, -1, 1,
			1, -1, -1, 1,
			1, -1, 1, 1,
			-1, -1, -1, 1,

			1, 1, 1, 1,
			1, -1, 1, 1,
			1, -1, -1, 1,
			1, 1, -1, 1,
			1, 1, 1, 1,
			1, -1, -1, 1,

			-1, 1, 1, 1,
			1, 1, 1, 1,
			1, 1, -1, 1,
			-1, 1, -1, 1,
			-1, 1, 1, 1,
			1, 1, -1, 1,

			-1, -1, 1, 1,
			-1, 1, 1, 1,
			-1, 1, -1, 1,
			-1, -1, -1, 1,
			-1, -1, 1, 1,
			-1, 1, -1, 1,

			1, 1, 1, 1,
			-1, 1, 1, 1,
			-1, -1, 1, 1,
			-1, -1, 1, 1,
			1, -1, 1, 1,
			1, 1, 1, 1,

			1, -1, -1, 1,
			-1, -1, -1, 1,
			-1, 1, -1, 1,
			1, 1, -1, 1,
			1, -1, -1, 1,
			-1, 1, -1, 1
		]);
	}
}