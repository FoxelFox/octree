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
	deferredTaaBindGroupA: GPUBindGroup;
	deferredTaaBindGroupB: GPUBindGroup;
	
	initialized: boolean;
	timer: RenderTimer;
	
	// G-buffer textures
	positionTexture: GPUTexture;
	normalTexture: GPUTexture;
	diffuseTexture: GPUTexture;
	depthTexture: GPUTexture;
	
	// TAA textures (ping-pong buffers)
	prevFrameTextureA: GPUTexture;
	prevFrameTextureB: GPUTexture;
	prevWorldPosTextureA: GPUTexture;
	prevWorldPosTextureB: GPUTexture;
	sampler: GPUSampler;
	
	// Current frame index for ping-pong
	frameIndex: number = 0;
	
	// Clean ping-pong approach with alternating bind groups
	getCurrentFrameTexture() {
		return this.frameIndex % 2 === 0 ? this.prevFrameTextureA : this.prevFrameTextureB;
	}
	
	getPreviousFrameTexture() {
		return this.frameIndex % 2 === 0 ? this.prevFrameTextureB : this.prevFrameTextureA;
	}
	
	getCurrentWorldPosTexture() {
		return this.frameIndex % 2 === 0 ? this.prevWorldPosTextureA : this.prevWorldPosTextureB;
	}
	
	getPreviousWorldPosTexture() {
		return this.frameIndex % 2 === 0 ? this.prevWorldPosTextureB : this.prevWorldPosTextureA;
	}
	
	getCurrentTaaBindGroup() {
		// When writing to A (even frame), read from B (use bind group B)
		// When writing to B (odd frame), read from A (use bind group A) 
		return this.frameIndex % 2 === 0 ? this.deferredTaaBindGroupB : this.deferredTaaBindGroupA;
	}

	constructor() {
		this.timer = new RenderTimer("block");
		this.createGBufferTextures();
		this.createTaaTextures();
		this.createSampler();
	}

	createSampler() {
		this.sampler = device.createSampler({
			magFilter: "linear",
			minFilter: "linear",
		});
	}

	createTaaTextures() {
		// Destroy existing TAA textures
		if (this.prevFrameTextureA) this.prevFrameTextureA.destroy();
		if (this.prevFrameTextureB) this.prevFrameTextureB.destroy();
		if (this.prevWorldPosTextureA) this.prevWorldPosTextureA.destroy();
		if (this.prevWorldPosTextureB) this.prevWorldPosTextureB.destroy();

		const size = {width: canvas.width, height: canvas.height};
		
		// Previous frame color textures (ping-pong buffers)
		this.prevFrameTextureA = device.createTexture({
			size,
			format: "bgra8unorm",
			usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
		});
		
		this.prevFrameTextureB = device.createTexture({
			size,
			format: "bgra8unorm",
			usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
		});

		// Previous world position textures (ping-pong buffers)
		this.prevWorldPosTextureA = device.createTexture({
			size,
			format: "rgba32float",
			usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
		});
		
		this.prevWorldPosTextureB = device.createTexture({
			size,
			format: "rgba32float",
			usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
		});
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
			usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
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
			this.createTaaTextures();
			// Need to recreate deferred bind groups with new textures
			this.createDeferredBindGroup();
			this.createDeferredTaaBindGroups();
		}
		
		// TAA bind groups are static, no need to recreate each frame

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

		// Pass 2: Deferred lighting with background and TAA
		const deferredPass = commandEncoder.beginRenderPass({
			label: "Block Deferred Lighting",
			colorAttachments: [
				{
					view: context.getCurrentTexture().createView(),
					loadOp: "clear",
					storeOp: "store",
					clearValue: {r: 0, g: 0, b: 0, a: 0},
				},
				{
					view: this.getCurrentFrameTexture().createView(),
					loadOp: this.frameIndex === 0 ? "clear" : "load",  // Clear first frame, load subsequent frames
					storeOp: "store",
					clearValue: {r: 0, g: 0, b: 0, a: 0},
				},
				{
					view: this.getCurrentWorldPosTexture().createView(),
					loadOp: this.frameIndex === 0 ? "clear" : "load",  // Clear first frame, load subsequent frames
					storeOp: "store", 
					clearValue: {r: 0, g: 0, b: 0, a: 0},
				},
			],
		});

		deferredPass.setPipeline(this.deferredPipeline);
		deferredPass.setBindGroup(0, this.deferredUniformBindGroup);
		deferredPass.setBindGroup(1, this.deferredBindGroup);
		deferredPass.setBindGroup(2, this.getCurrentTaaBindGroup());
		deferredPass.draw(6); // Full-screen quad

		deferredPass.end();

		// Just increment frame index for ping-pong logic
		this.frameIndex++;

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

	createDeferredTaaBindGroups() {
		// Bind group A: reads from texture A (when we write to texture B)  
		this.deferredTaaBindGroupA = device.createBindGroup({
			label: "Deferred TAA Textures A",
			layout: this.deferredPipeline.getBindGroupLayout(2),
			entries: [
				{binding: 0, resource: this.prevFrameTextureA.createView()},
				{binding: 1, resource: this.sampler},
				{binding: 2, resource: this.prevWorldPosTextureA.createView()},
			],
		});
		
		// Bind group B: reads from texture B (when we write to texture A)
		this.deferredTaaBindGroupB = device.createBindGroup({
			label: "Deferred TAA Textures B",
			layout: this.deferredPipeline.getBindGroupLayout(2),
			entries: [
				{binding: 0, resource: this.prevFrameTextureB.createView()},
				{binding: 1, resource: this.sampler},
				{binding: 2, resource: this.prevWorldPosTextureB.createView()},
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
				cullMode: "back"
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
					{format: "bgra8unorm"}, // Canvas
					{format: "bgra8unorm"}, // Previous frame texture
					{format: "rgba32float"}, // Previous world position texture
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
		this.createDeferredTaaBindGroups();

		this.initialized = true;
	}
}
