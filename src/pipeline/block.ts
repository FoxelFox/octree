import {canvas, context, contextUniform, device} from "../index";
import shader from "./block.wgsl" with {type: "text"};
import deferredShader from "./block_deferred.wgsl" with {type: "text"};
import {Cull} from "./cull";
import {Mesh} from "./mesh";
import {RenderTimer} from "./timing";

export class Block {
	// input
	mesh: Mesh;
	cull: Cull;

	// G-buffer pass
	gBufferPipeline: GPURenderPipeline;
	gBufferBindGroup: GPUBindGroup;
	gBufferUniformBindGroup: GPUBindGroup;

	// Deferred lighting pass
	deferredPipeline: GPURenderPipeline;
	deferredBindGroup: GPUBindGroup;
	deferredUniformBindGroup: GPUBindGroup;

	initialized: boolean;
	timer: RenderTimer;

	// G-buffer textures
	positionTexture: GPUTexture;
	normalTexture: GPUTexture;
	diffuseTexture: GPUTexture;
	depthTexture: GPUTexture;


	constructor() {
		this.timer = new RenderTimer("block");
		this.createGBufferTextures();
	}


	createGBufferTextures() {
		// Destroy existing textures
		if (this.positionTexture) this.positionTexture.destroy();
		if (this.normalTexture) this.normalTexture.destroy();
		if (this.diffuseTexture) this.diffuseTexture.destroy();
		if (this.depthTexture) this.depthTexture.destroy();

		const size = {width: canvas.width, height: canvas.height};

		// Position texture (RGBA32Float for high precision world positions)
		this.positionTexture = device.createTexture({
			size,
			format: "rgba32float",
			usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
		});

		// Normal texture (RGBA16Float is sufficient for normals)
		this.normalTexture = device.createTexture({
			size,
			format: "rgba16float",
			usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
		});

		// Diffuse color texture (RGBA8Unorm for colors)
		this.diffuseTexture = device.createTexture({
			size,
			format: "rgba8unorm",
			usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
		});

		// Depth texture
		this.depthTexture = device.createTexture({
			size,
			format: "depth24plus",
			usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
		});
	}

	update(commandEncoder: GPUCommandEncoder) {
		if (!this.initialized) {
			this.init();
		}

		// Recreate G-buffer textures if canvas size changed
		if (
			this.depthTexture.width !== canvas.width ||
			this.depthTexture.height !== canvas.height
		) {
			this.createGBufferTextures();
			// Need to recreate deferred bind groups with new textures
			this.createDeferredBindGroup();
		}

		// Pass 1: G-buffer generation
		const gBufferPass = commandEncoder.beginRenderPass({
			label: "Block G-Buffer",
			colorAttachments: [
				{
					view: this.positionTexture.createView(),
					loadOp: "clear",
					storeOp: "store",
					clearValue: {r: 0, g: 0, b: 0, a: 0},
				},
				{
					view: this.normalTexture.createView(),
					loadOp: "clear",
					storeOp: "store",
					clearValue: {r: 0, g: 0, b: 0, a: 0},
				},
				{
					view: this.diffuseTexture.createView(),
					loadOp: "clear",
					storeOp: "store",
					clearValue: {r: 0, g: 0, b: 0, a: 0},
				},
			],
			depthStencilAttachment: {
				view: this.depthTexture.createView(),
				depthClearValue: 1.0,
				depthLoadOp: "clear",
				depthStoreOp: "store",
			},
			timestampWrites: this.timer.getTimestampWrites(),
		});

		gBufferPass.setPipeline(this.gBufferPipeline);
		gBufferPass.setBindGroup(0, this.gBufferBindGroup);
		gBufferPass.setBindGroup(1, this.gBufferUniformBindGroup);

		// Draw instances for each mesh chunk (only if culling data is ready)
		if (this.cull.count > 0 && this.cull.indices && this.cull.indices.length > 0) {
			for (let i = 0; i < this.cull.count && i < this.cull.indices.length; ++i) {
				const meshIndex = this.cull.indices[i];
				if (typeof meshIndex === 'number' && isFinite(meshIndex)) {
					gBufferPass.drawIndirect(this.mesh.commands, meshIndex * 16);
				}
			}
		}

		gBufferPass.end();

		// Pass 2: Deferred lighting with background
		const deferredPass = commandEncoder.beginRenderPass({
			label: "Block Deferred Lighting",
			colorAttachments: [
				{
					view: context.getCurrentTexture().createView(),
					loadOp: "clear",
					storeOp: "store",
					clearValue: {r: 0, g: 0, b: 0, a: 0},
				},
			],
		});

		deferredPass.setPipeline(this.deferredPipeline);
		deferredPass.setBindGroup(0, this.deferredUniformBindGroup);
		deferredPass.setBindGroup(1, this.deferredBindGroup);
		deferredPass.draw(6); // Full-screen quad

		deferredPass.end();

		this.timer.resolveTimestamps(commandEncoder);
	}

	afterUpdate() {
		this.timer.readTimestamps();
	}

	get renderTime(): number {
		return this.timer.renderTime;
	}

	createDeferredBindGroup() {
		this.deferredBindGroup = device.createBindGroup({
			label: "Deferred G-Buffer Textures",
			layout: this.deferredPipeline.getBindGroupLayout(1),
			entries: [
				{binding: 0, resource: this.positionTexture.createView()},
				{binding: 1, resource: this.normalTexture.createView()},
				{binding: 2, resource: this.diffuseTexture.createView()},
				{binding: 3, resource: this.depthTexture.createView()},
			],
		});
	}


	init() {
		// Create G-buffer pipeline
		this.gBufferPipeline = device.createRenderPipeline({
			label: "Block G-Buffer",
			layout: "auto",
			fragment: {
				module: device.createShaderModule({
					label: "Block G-Buffer Fragment Shader",
					code: shader,
				}),
				targets: [
					{format: "rgba32float"}, // Position
					{format: "rgba16float"}, // Normal
					{format: "rgba8unorm"},  // Diffuse
				],
			},
			vertex: {
				module: device.createShaderModule({
					label: "Block G-Buffer Vertex Shader",
					code: shader,
				}),
			},
			primitive: {
				topology: "triangle-list",
				cullMode: "front"
			},
			depthStencil: {
				depthWriteEnabled: true,
				depthCompare: "less",
				format: "depth24plus",
			},
		});

		// Create deferred lighting pipeline
		this.deferredPipeline = device.createRenderPipeline({
			label: "Block Deferred Lighting",
			layout: "auto",
			fragment: {
				module: device.createShaderModule({
					label: "Block Deferred Fragment Shader",
					code: deferredShader,
				}),
				targets: [
					{format: "bgra8unorm"}, // Canvas only
				],
			},
			vertex: {
				module: device.createShaderModule({
					label: "Block Deferred Vertex Shader",
					code: deferredShader,
				}),
			},
			primitive: {
				topology: "triangle-list",
			},
		});

		// Create G-buffer bind groups
		this.gBufferBindGroup = device.createBindGroup({
			label: "Block G-Buffer Meshes",
			layout: this.gBufferPipeline.getBindGroupLayout(0),
			entries: [{binding: 0, resource: this.mesh.meshes}],
		});

		this.gBufferUniformBindGroup = device.createBindGroup({
			label: "Block G-Buffer Context",
			layout: this.gBufferPipeline.getBindGroupLayout(1),
			entries: [{binding: 0, resource: contextUniform.uniformBuffer}],
		});

		// Create deferred bind groups
		this.deferredUniformBindGroup = device.createBindGroup({
			label: "Deferred Context",
			layout: this.deferredPipeline.getBindGroupLayout(0),
			entries: [{binding: 0, resource: contextUniform.uniformBuffer}],
		});

		this.createDeferredBindGroup();

		this.initialized = true;
	}
}
