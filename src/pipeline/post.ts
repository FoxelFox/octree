import {canvas, context, contextUniform, device} from "../index";
import {BlueNoise} from "./bluenoise";
import {DistanceField} from "./distance_field";
import shader from "./post.wgsl" with {type: "text"};
import {Noise} from "./noise";
import {RenderTimer} from "./timing";

export class Post {
	pipeline: GPURenderPipeline;

	uniformBindGroup: GPUBindGroup;
	noise: Noise;
	blueNoise: BlueNoise;
	distanceField: DistanceField;

	frameBuffers: GPUTexture[] = [];
	worldPosBuffers: GPUTexture[] = [];
	frameBufferBindgroups: GPUBindGroup[] = [];
	sampler: GPUSampler;

	timer: RenderTimer;
	frame = 0;

	constructor() {
		// Create explicit bind group layouts
		const uniformBindGroupLayout = device.createBindGroupLayout({
			entries: [
				{
					binding: 0,
					visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
					buffer: {type: 'uniform'}
				},
				{
					binding: 1,
					visibility: GPUShaderStage.FRAGMENT,
					buffer: {type: 'read-only-storage'}
				},
				{
					binding: 2,
					visibility: GPUShaderStage.FRAGMENT,
					buffer: {type: 'read-only-storage'}
				}
			]
		});

		const textureBindGroupLayout = device.createBindGroupLayout({
			entries: [
				{
					binding: 0,
					visibility: GPUShaderStage.FRAGMENT,
					texture: {sampleType: 'float'}
				},
				{
					binding: 1,
					visibility: GPUShaderStage.FRAGMENT,
					sampler: {}
				},
				{
					binding: 2,
					visibility: GPUShaderStage.FRAGMENT,
					texture: {sampleType: 'unfilterable-float'}
				},
				{
					binding: 3,
					visibility: GPUShaderStage.FRAGMENT,
					texture: {sampleType: 'float'}
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

		// Create timing utility
		this.timer = new RenderTimer('post');

		// Initialize blue noise
		this.blueNoise = new BlueNoise();

		this.resizeFrameBuffer();
	}

	init() {
		if (!this.noise) {
			throw new Error('Noise must be set before calling init()');
		}

		if (!this.distanceField) {
			throw new Error('DistanceField must be set before calling init()');
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
				},
				{
					binding: 2,
					resource: {buffer: this.distanceField.getDistanceFieldBuffer()}
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
			timestampWrites: this.timer.getTimestampWrites(),
		});
		passEncoder.setPipeline(this.pipeline);
		passEncoder.setBindGroup(0, this.uniformBindGroup);
		passEncoder.setBindGroup(1, this.frameBufferBindgroups[this.frame % 2]);
		passEncoder.draw(6);
		passEncoder.end();

		this.timer.resolveTimestamps(commandEncoder);
		this.frame++;
	}

	afterUpdate() {
		this.timer.readTimestamps();
	}

	get renderTime(): number {
		return this.timer.renderTime;
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
				{binding: 2, resource: this.worldPosBuffers[0].createView()},
				{binding: 3, resource: this.blueNoise.getTextureView()}
			]
		}));

		this.frameBufferBindgroups.push(device.createBindGroup({
			layout: this.pipeline.getBindGroupLayout(1),
			entries: [
				{binding: 0, resource: this.frameBuffers[1].createView()},
				{binding: 1, resource: this.sampler},
				{binding: 2, resource: this.worldPosBuffers[1].createView()},
				{binding: 3, resource: this.blueNoise.getTextureView()}
			]
		}));
	}
}