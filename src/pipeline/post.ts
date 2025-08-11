import {device, context, canvas, contextUniform} from "../index";
import {Noise} from "./noise";
import shader from "./post.wgsl" with {type: "text"};

export class Post {
	pipeline: GPURenderPipeline;

	uniformBindGroup: GPUBindGroup;
	noise: Noise;

	frameBuffers: GPUTexture[] = [];
	worldPosBuffers: GPUTexture[] = [];
	frameBufferBindgroups: GPUBindGroup[] = [];
	sampler: GPUSampler;

	// timing
	querySet: GPUQuerySet;
	queryBuffer: GPUBuffer;
	queryReadbackBuffer: GPUBuffer;
	isReadingTiming: boolean = false;
	renderTime: number = 0;
	lastTimingFrame: number = 0;

	frame = 0;

	constructor() {
		// Create explicit bind group layouts
		const uniformBindGroupLayout = device.createBindGroupLayout({
			entries: [
				{
					binding: 0,
					visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
					buffer: { type: 'uniform' }
				},
				{
					binding: 1,
					visibility: GPUShaderStage.FRAGMENT,
					buffer: { type: 'storage' }
				}
			]
		});

		const textureBindGroupLayout = device.createBindGroupLayout({
			entries: [
				{
					binding: 0,
					visibility: GPUShaderStage.FRAGMENT,
					texture: { sampleType: 'float' }
				},
				{
					binding: 1,
					visibility: GPUShaderStage.FRAGMENT,
					sampler: {}
				},
				{
					binding: 2,
					visibility: GPUShaderStage.FRAGMENT,
					texture: { sampleType: 'unfilterable-float' }
				}
			]
		});

		const pipelineLayout = device.createPipelineLayout({
			bindGroupLayouts: [uniformBindGroupLayout, textureBindGroupLayout]
		});

		this.pipeline = device.createRenderPipeline({
			layout: pipelineLayout,
			vertex: {
				module: device.createShaderModule({
					code: shader
				}),
				entryPoint: 'main_vs',
			},
			fragment: {
				module: device.createShaderModule({
					code: shader
				}),
				entryPoint: 'main_fs',
				targets: [
					{
						format: 'bgra8unorm',
						blend: {
							color: {
								srcFactor: 'one',
								dstFactor: 'one-minus-src-alpha'
							},
							alpha: {
								srcFactor: 'one',
								dstFactor: 'one-minus-src-alpha'
							}
						}
					},
					{format: 'bgra8unorm'},
			{format: 'rgba32float'}
				]
			},
			primitive: {
				topology: 'triangle-list'
			}
		});

		// Create timestamp query set for precise timing
		this.querySet = device.createQuerySet({
			type: 'timestamp',
			count: 2, // start and end timestamps
		});

		this.queryBuffer = device.createBuffer({
			size: 16, // 2 timestamps * 8 bytes each
			usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
		});

		this.queryReadbackBuffer = device.createBuffer({
			size: 16,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
		});

		this.resizeFrameBuffer();
	}

	init() {
		if (!this.noise) {
			throw new Error('Noise must be set before calling init()');
		}

		this.uniformBindGroup = device.createBindGroup({
			layout: this.pipeline.getBindGroupLayout(0),
			entries: [
				{
					binding: 0,
					resource: {buffer: contextUniform.uniformBuffer}
				},
				{
					binding: 1,
					resource: {buffer: this.noise.nodesBuffer}
				}
			]
		});
	}

	update(commandEncoder: GPUCommandEncoder) {
		if (!this.uniformBindGroup) {
			this.init();
		}

		if (this.frameBuffers[0].width !== canvas.width || this.frameBuffers[0].height !== canvas.height) {
			this.resizeFrameBuffer();
		}

		// Only do timing every 60 frames to avoid conflicts
		const shouldMeasureTiming = !this.isReadingTiming && (this.frame - this.lastTimingFrame) > 60;

		const passEncoder = commandEncoder.beginRenderPass({
			colorAttachments: [{
				view: context.getCurrentTexture().createView(),
				loadOp: 'load',
				storeOp: 'store',
			}, {
				view: this.frameBuffers[(this.frame + 1) % 2].createView(),
				loadOp: 'load',
				storeOp: 'store',
			}, {
				view: this.worldPosBuffers[(this.frame + 1) % 2].createView(),
				loadOp: 'load',
				storeOp: 'store',
			}],
			timestampWrites: shouldMeasureTiming ? {
				querySet: this.querySet,
				beginningOfPassWriteIndex: 0,
				endOfPassWriteIndex: 1,
			} : undefined,
		});
		passEncoder.setPipeline(this.pipeline);
		passEncoder.setBindGroup(0, this.uniformBindGroup);
		passEncoder.setBindGroup(1, this.frameBufferBindgroups[this.frame % 2]);
		passEncoder.draw(6);
		passEncoder.end();

		// Only resolve timestamp queries when measuring
		if (shouldMeasureTiming) {
			commandEncoder.resolveQuerySet(this.querySet, 0, 2, this.queryBuffer, 0);
			commandEncoder.copyBufferToBuffer(this.queryBuffer, 0, this.queryReadbackBuffer, 0, 16);
			this.lastTimingFrame = this.frame;
		}

		this.frame++
	}

	afterUpdate() {
		if (!this.isReadingTiming && (this.frame - this.lastTimingFrame) === 1) {
			this.isReadingTiming = true;
			this.queryReadbackBuffer.mapAsync(GPUMapMode.READ).then(() => {
				const times = new BigUint64Array(this.queryReadbackBuffer.getMappedRange());
				const startTime = times[0];
				const endTime = times[1];
				
				// Only update if we have valid timestamps
				if (startTime > 0n && endTime > 0n && endTime >= startTime) {
					const duration = endTime - startTime;
					this.renderTime = Number(duration) / 1_000_000; // Convert to milliseconds
				}
				// Keep previous renderTime value if timestamps are invalid
				
				this.queryReadbackBuffer.unmap();
				this.isReadingTiming = false;
			}).catch(() => {
				// Handle mapping failure gracefully
				this.isReadingTiming = false;
			});
		}
	}

	resizeFrameBuffer() {
		if (this.frameBuffers.length) {
			this.frameBuffers[0].destroy();
			this.frameBuffers[1].destroy();
			this.worldPosBuffers[0]?.destroy();
			this.worldPosBuffers[1]?.destroy();

			this.frameBuffers.length = 0;
			this.worldPosBuffers.length = 0;
			this.frameBufferBindgroups.length = 0;
		}

		for (let i = 0; i < 2; i++) {
			this.frameBuffers.push(device.createTexture({
				size: {width: canvas.width, height: canvas.height},
				format: navigator.gpu.getPreferredCanvasFormat(),
				usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
			}));
			
			this.worldPosBuffers.push(device.createTexture({
				size: {width: canvas.width, height: canvas.height},
				format: 'rgba32float',
				usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
			}));
		}

		if (!this.sampler) {
			this.sampler = device.createSampler({
				magFilter: 'linear',
				minFilter: 'linear'
			});
		}

		this.frameBufferBindgroups.push(device.createBindGroup({
			layout: this.pipeline.getBindGroupLayout(1),
			entries: [
				{binding: 0, resource: this.frameBuffers[0].createView()},
				{binding: 1, resource: this.sampler},
				{binding: 2, resource: this.worldPosBuffers[0].createView()}
			]
		}));

		this.frameBufferBindgroups.push(device.createBindGroup({
			layout: this.pipeline.getBindGroupLayout(1),
			entries: [
				{binding: 0, resource: this.frameBuffers[1].createView()},
				{binding: 1, resource: this.sampler},
				{binding: 2, resource: this.worldPosBuffers[1].createView()}
			]
		}));
	}
}