import {canvas, context, contextUniform, device, gridSize, compression} from "../../index";
import shader from "./block.wgsl" with {type: "text"};
import deferredShader from "./block_deferred.wgsl" with {type: "text"};
import spaceBackgroundShader from "../generation/space_background.wgsl" with {type: "text"};
import {Cull} from "./cull";
import {Mesh} from "../generation/mesh";
import {Light} from "./light";
import {RenderTimer} from "../timing";

export class Block {
	// input
	mesh: Mesh;
	cull: Cull;
	light: Light;

	// G-buffer pass
	gBufferPipeline: GPURenderPipeline;
	gBufferBindGroup: GPUBindGroup;
	gBufferUniformBindGroup: GPUBindGroup;

	// Deferred lighting pass
	deferredPipeline: GPURenderPipeline;
	deferredBindGroup: GPUBindGroup;
	deferredUniformBindGroup: GPUBindGroup;
	deferredSpaceBindGroup: GPUBindGroup;
	deferredLightBindGroup: GPUBindGroup;

	// Space background
	spaceBackgroundTexture: GPUTexture;
	spaceBackgroundPipeline: GPUComputePipeline;
	spaceBackgroundBindGroup: GPUBindGroup;

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
		this.createSpaceBackgroundTexture();
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

	createSpaceBackgroundTexture() {
		// Destroy existing texture
		if (this.spaceBackgroundTexture) this.spaceBackgroundTexture.destroy();

		// Create space background texture (2048x1024 for good quality)
		this.spaceBackgroundTexture = device.createTexture({
			size: { width: 2048, height: 1024 },
			format: "rgba8unorm",
			usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
		});
	}

	generateSpaceBackground(commandEncoder: GPUCommandEncoder) {
		// Generate space background texture once using compute shader
		const computePass = commandEncoder.beginComputePass({
			label: "Generate Space Background",
		});

		computePass.setPipeline(this.spaceBackgroundPipeline);
		computePass.setBindGroup(0, this.spaceBackgroundBindGroup);
		computePass.dispatchWorkgroups(
			Math.ceil(2048 / 8), // workgroup size is 8x8
			Math.ceil(1024 / 8)
		);
		computePass.end();
	}

	update(commandEncoder: GPUCommandEncoder) {
		if (!this.initialized) {
			this.init();
			// Generate space background texture once after initialization
			this.generateSpaceBackground(commandEncoder);
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
			const maxMeshIndex = Math.pow(gridSize / compression, 3) - 1;
			for (let i = 0; i < this.cull.count && i < this.cull.indices.length; ++i) {
				const meshIndex = this.cull.indices[i];
				if (typeof meshIndex === 'number' && isFinite(meshIndex) && meshIndex <= maxMeshIndex) {
					gBufferPass.drawIndirect(this.mesh.commands, meshIndex * 16);
				} else if (meshIndex > maxMeshIndex) {
					console.warn(`Mesh index ${meshIndex} exceeds maximum ${maxMeshIndex}, skipping`);
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
		deferredPass.setBindGroup(2, this.deferredSpaceBindGroup);
		deferredPass.setBindGroup(3, this.deferredLightBindGroup);
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
		// Create space background compute pipeline
		this.spaceBackgroundPipeline = device.createComputePipeline({
			label: "Space Background Generation",
			layout: "auto",
			compute: {
				module: device.createShaderModule({
					label: "Space Background Compute Shader",
					code: spaceBackgroundShader,
				}),
			},
		});

		// Create space background bind group
		this.spaceBackgroundBindGroup = device.createBindGroup({
			label: "Space Background",
			layout: this.spaceBackgroundPipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: this.spaceBackgroundTexture.createView()},
			],
		});

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

		// Create deferred space bind group for space background texture
		this.deferredSpaceBindGroup = device.createBindGroup({
			label: "Deferred Space Background",
			layout: this.deferredPipeline.getBindGroupLayout(2),
			entries: [
				{binding: 0, resource: this.spaceBackgroundTexture.createView()},
			],
		});

		// Create deferred light bind group for voxel lighting data
		this.deferredLightBindGroup = device.createBindGroup({
			label: "Deferred Light Data",
			layout: this.deferredPipeline.getBindGroupLayout(3),
			entries: [
				{binding: 0, resource: {buffer: this.light.getLightBuffer()}},
			],
		});

		this.initialized = true;
	}
}
