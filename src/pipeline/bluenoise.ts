import {device} from "../index";
import shader from "./bluenoise.wgsl" with {type: "text"};

export class BlueNoise {
    texture: GPUTexture;
    computePipeline: GPUComputePipeline;
    bindGroup: GPUBindGroup;
    
    private static readonly BLUE_NOISE_SIZE = 64;

    constructor() {
        // Create the blue noise texture
        this.texture = device.createTexture({
            size: {
                width: BlueNoise.BLUE_NOISE_SIZE,
                height: BlueNoise.BLUE_NOISE_SIZE,
                depthOrArrayLayers: 1
            },
            format: 'rgba8unorm',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
        });

        // Create compute pipeline
        this.computePipeline = device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: device.createShaderModule({
                    code: shader
                }),
                entryPoint: 'main'
            }
        });

        // Create bind group
        this.bindGroup = device.createBindGroup({
            layout: this.computePipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: this.texture.createView()
                }
            ]
        });

        // Generate the blue noise texture
        this.generate();
    }

    private generate() {
        const commandEncoder = device.createCommandEncoder();
        const computePass = commandEncoder.beginComputePass();
        
        computePass.setPipeline(this.computePipeline);
        computePass.setBindGroup(0, this.bindGroup);
        
        // Dispatch with workgroups to cover the entire texture
        const workgroupsX = Math.ceil(BlueNoise.BLUE_NOISE_SIZE / 8);
        const workgroupsY = Math.ceil(BlueNoise.BLUE_NOISE_SIZE / 8);
        computePass.dispatchWorkgroups(workgroupsX, workgroupsY);
        
        computePass.end();
        device.queue.submit([commandEncoder.finish()]);
    }

    getTextureView(): GPUTextureView {
        return this.texture.createView();
    }

    destroy() {
        this.texture.destroy();
    }
}