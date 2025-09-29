import {canvas, compression, context, contextUniform, device, gridSize} from "../../index";
import shader from "./block.wgsl" with {type: "text"};
import deferredShader from "./block_deferred.wgsl" with {type: "text"};
import spaceBackgroundShader from "../generation/space_background.wgsl" with {type: "text"};
import {RenderTimer} from "../timing";
import {Chunk} from "../../chunk/chunk";

export class Block {

	// G-buffer pass
	gBufferPipeline: GPURenderPipeline;
	gBufferBindGroups = new Map<Chunk, GPUBindGroup>();
	gBufferUniformBindGroup: GPUBindGroup;

	// Deferred lighting pass
	deferredPipeline: GPURenderPipeline;
	deferredBindGroup: GPUBindGroup;
	deferredUniformBindGroups = new Map<Chunk, GPUBindGroup>();
	deferredSpaceBindGroup: GPUBindGroup;
	deferredLightBindGroups = new Map<Chunk, GPUBindGroup>();
	chunkWorldPosBuffers = new Map<Chunk, GPUBuffer>();
	private lightBufferRefs = new Map<Chunk, GPUBuffer>();

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
					{
						format: "bgra8unorm",
						blend: {
							color: {
								srcFactor: "src-alpha",
								dstFactor: "one-minus-src-alpha",
								operation: "add",
							},
							alpha: {
								srcFactor: "one",
								dstFactor: "one-minus-src-alpha",
								operation: "add",
							},
						},
					},
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


		this.gBufferUniformBindGroup = device.createBindGroup({
			label: "Block G-Buffer Context",
			layout: this.gBufferPipeline.getBindGroupLayout(1),
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
	}

	get renderTime(): number {
		return this.timer.renderTime;
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
			size: {width: 2048, height: 1024},
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

		this.initialized = true;
	}

	update(commandEncoder: GPUCommandEncoder, chunks: Chunk[]) {
		if (!this.initialized) {
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

		// Update deferred light bind groups for all chunks
		for (const chunk of chunks) {
			this.updateDeferredLightBindGroup(chunk);
		}

		// Pass 1: G-buffer generation for all chunks
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

		// Render all chunks into G-buffer
		for (const chunk of chunks) {
			gBufferPass.setBindGroup(0, this.gBufferBindGroups.get(chunk));
			gBufferPass.setBindGroup(1, this.gBufferUniformBindGroup);

			// Draw instances for each mesh chunk (only if culling data is ready)
			if (chunk.indices && chunk.indices.length > 0) {
				const maxMeshIndex = Math.pow(gridSize / compression, 3) - 1;
				for (let i = 0; i < chunk.indices.length; ++i) {
					const meshIndex = chunk.indices[i];
					if (typeof meshIndex === 'number' && isFinite(meshIndex) && meshIndex <= maxMeshIndex) {
						gBufferPass.drawIndirect(chunk.commands, meshIndex * 16);
					} else if (meshIndex > maxMeshIndex) {
						console.warn(`Mesh index ${meshIndex} exceeds maximum ${maxMeshIndex}, skipping`);
					}
				}
			}
		}

		gBufferPass.end();

		// Pass 2: Deferred lighting with background
		// We need to do this for each chunk because each chunk has different light data
		// Use "load" after the first chunk to preserve previous results
		let isFirstChunk = true;
		for (const chunk of chunks) {
			const deferredPass = commandEncoder.beginRenderPass({
				label: `Block Deferred Lighting - Chunk ${chunk.id}`,
				colorAttachments: [
					{
						view: context.getCurrentTexture().createView(),
						loadOp: isFirstChunk ? "clear" : "load",
						storeOp: "store",
						clearValue: {r: 0, g: 0, b: 0, a: 0},
					},
				],
			});

			deferredPass.setPipeline(this.deferredPipeline);
			deferredPass.setBindGroup(0, this.deferredUniformBindGroups.get(chunk)!);
			deferredPass.setBindGroup(1, this.deferredBindGroup);
			deferredPass.setBindGroup(2, this.deferredSpaceBindGroup);
			deferredPass.setBindGroup(3, this.deferredLightBindGroups.get(chunk)!);
			deferredPass.draw(6); // Full-screen quad

			deferredPass.end();
			isFirstChunk = false;
		}

		this.timer.resolveTimestamps(commandEncoder);
	}

	afterUpdate() {
		this.timer.readTimestamps();
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

	registerChunk(chunk: Chunk) {

		// Create G-buffer bind groups
		const gBufferBindGroup = device.createBindGroup({
			label: "Block G-Buffer Meshes",
			layout: this.gBufferPipeline.getBindGroupLayout(0),
			entries: [
				{binding: 1, resource: {buffer: chunk.vertices}},
				{binding: 2, resource: {buffer: chunk.normals}},
				{binding: 3, resource: {buffer: chunk.colors}},
			],
		});

		this.gBufferBindGroups.set(chunk, gBufferBindGroup);

		// Create chunk world position buffer
		const chunkWorldPosBuffer = device.createBuffer({
			size: 16, // vec3<i32> + padding = 16 bytes
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		// Write chunk world position (in voxels)
		const chunkWorldPosData = new Int32Array([
			chunk.position[0] * gridSize,
			chunk.position[1] * gridSize,
			chunk.position[2] * gridSize,
			0 // padding
		]);
		device.queue.writeBuffer(chunkWorldPosBuffer, 0, chunkWorldPosData);

		// Create deferred uniform bind group with chunk world position
		const deferredUniformBindGroup = device.createBindGroup({
			label: "Deferred Context",
			layout: this.deferredPipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: contextUniform.uniformBuffer},
				{binding: 1, resource: {buffer: chunkWorldPosBuffer}}
			],
		});

		this.deferredUniformBindGroups.set(chunk, deferredUniformBindGroup);
		this.chunkWorldPosBuffers.set(chunk, chunkWorldPosBuffer);
		this.updateDeferredLightBindGroup(chunk);
	}

	private updateDeferredLightBindGroup(chunk: Chunk) {
		const current = this.lightBufferRefs.get(chunk);
		if (current === chunk.light && this.deferredLightBindGroups.has(chunk)) {
			return;
		}

		const deferredLightBindGroup = device.createBindGroup({
			label: "Deferred Light Data",
			layout: this.deferredPipeline.getBindGroupLayout(3),
			entries: [
				{binding: 0, resource: {buffer: chunk.light}},
			],
		});

		this.deferredLightBindGroups.set(chunk, deferredLightBindGroup);
		this.lightBufferRefs.set(chunk, chunk.light);
	}
}
