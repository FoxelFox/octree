import compactShader from './compact.wgsl';
import {device, maxDepth, worstCaseMaxNodes} from "../index";
import {Noise} from "./noise";

export class Compact {
    public pipeline: GPUComputePipeline;
    public bindGroup: GPUBindGroup;
    public compactNodesBuffer: GPUBuffer;
    public compactNodeCounterBuffer: GPUBuffer;
    public compactNodeCounterReadbackBuffer: GPUBuffer;
    public leafNodeCounterBuffer: GPUBuffer;
    public leafNodeCounterReadbackBuffer: GPUBuffer;

    // timing
    querySet: GPUQuerySet;
    queryBuffer: GPUBuffer;
    queryReadbackBuffer: GPUBuffer;
    isReadingTiming: boolean = false;
    compactTime: number = 0;

    constructor(noise: Noise) {
        this.compactNodesBuffer = device.createBuffer({
            size: worstCaseMaxNodes * 8 * 2, // CompactNode is 8 bytes
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });

        this.compactNodeCounterBuffer = device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });

        this.compactNodeCounterReadbackBuffer = device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        this.leafNodeCounterBuffer = device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });

        this.leafNodeCounterReadbackBuffer = device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        const bindGroupLayout = device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'read-only-storage' },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'storage' },
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'storage' },
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'storage' },
                },
            ],
        });

        this.bindGroup = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: { buffer: noise.nodesBuffer },
                },
                {
                    binding: 1,
                    resource: { buffer: this.compactNodesBuffer },
                },
                {
                    binding: 2,
                    resource: { buffer: this.compactNodeCounterBuffer },
                },
                {
                    binding: 3,
                    resource: { buffer: this.leafNodeCounterBuffer },
                },
            ],
        });

        this.pipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: compactShader }),
                entryPoint: 'main',
            },
        });

        this.querySet = device.createQuerySet({
            type: 'timestamp',
            count: 2,
        });

        this.queryBuffer = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        });

        this.queryReadbackBuffer = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
    }

    update(commandEncoder: GPUCommandEncoder) {
        // Initialize counters to 0 before running the compute shader
        const zeroBuffer = new Uint32Array([0]);
        device.queue.writeBuffer(this.compactNodeCounterBuffer, 0, zeroBuffer);
        device.queue.writeBuffer(this.leafNodeCounterBuffer, 0, zeroBuffer);

        const pass = commandEncoder.beginComputePass({
            timestampWrites: {
                querySet: this.querySet,
                beginningOfPassWriteIndex: 0,
                endOfPassWriteIndex: 1,
            }
        });
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.dispatchWorkgroups(1, 1, 1);
        pass.end();

        commandEncoder.resolveQuerySet(this.querySet, 0, 2, this.queryBuffer, 0);
        commandEncoder.copyBufferToBuffer(this.queryBuffer, 0, this.queryReadbackBuffer, 0, 16);
        commandEncoder.copyBufferToBuffer(this.compactNodeCounterBuffer, 0, this.compactNodeCounterReadbackBuffer, 0, 4);
        commandEncoder.copyBufferToBuffer(this.leafNodeCounterBuffer, 0, this.leafNodeCounterReadbackBuffer, 0, 4);
    }
    
    async readback() {
        if (this.isReadingTiming) return;
        this.isReadingTiming = true;

		await this.queryReadbackBuffer.mapAsync(GPUMapMode.READ);
		await this.compactNodeCounterReadbackBuffer.mapAsync(GPUMapMode.READ);
		await this.leafNodeCounterReadbackBuffer.mapAsync(GPUMapMode.READ);

        try {

            const times = new BigUint64Array(this.queryReadbackBuffer.getMappedRange());
            const startTime = times[0];
            const endTime = times[1];
            if (startTime > 0n && endTime > 0n && endTime >= startTime) {
                const duration = endTime - startTime;
                this.compactTime = Number(duration) / 1_000_000;
            }
            this.queryReadbackBuffer.unmap();
        } catch (e) {
            console.error('Error reading timing data:', e);
        }
        
        try {

            const compactCountData = new Uint32Array(this.compactNodeCounterReadbackBuffer.getMappedRange());
            const compactNodeCount = compactCountData[0];
            console.log('Compact node count:', compactNodeCount);
            this.compactNodeCounterReadbackBuffer.unmap();
        } catch (e) {
            console.error('Error reading compact node count:', e);
        }

        try {

            const leafCountData = new Uint32Array(this.leafNodeCounterReadbackBuffer.getMappedRange());
            const leafNodeCount = leafCountData[0];
            console.log('Leaf node count:', leafNodeCount);
            this.leafNodeCounterReadbackBuffer.unmap();
        } catch (e) {
            console.error('Error reading leaf node count:', e);
        }
        
        this.isReadingTiming = false;
    }

    afterUpdate() {}
}