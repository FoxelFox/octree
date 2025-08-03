import { contextUniform, device } from "..";
import shader from "./octree.wgsl" with {type: "text"};

export class Octree {
  
  octreeBuffer: GPUBuffer
  maxDepth: number = 1
  uniformBuffer: GPUBuffer
  pointerBuffer: GPUBuffer
  pipeline: GPUComputePipeline
  bindGroup0: GPUBindGroup
  bindGroup1: GPUBindGroup

  constructor() {
    this.octreeBuffer = device.createBuffer({
      label: "Octree",
      size: this.worstCaseMaxNodes * 10 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });

    this.uniformBuffer = device.createBuffer({
      label: "Octree Uniform",
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    this.pointerBuffer = device.createBuffer({
      label: "Octree Pointer",
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });

    device.queue.writeBuffer(this.uniformBuffer, 0, new Uint32Array([this.gridSize]));
    device.queue.writeBuffer(this.pointerBuffer, 0, new Uint32Array([0]));

    this.pipeline = device.createComputePipeline({
      label: "Octree",
      layout: "auto",
      compute: {
        module: device.createShaderModule({
          code: shader,
        }),
        entryPoint: "main",
      },
    });

    this.bindGroup0 = device.createBindGroup({
      label: "Octree",
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        {binding: 0, resource: this.uniformBuffer},
        {binding: 1, resource: this.octreeBuffer},
        {binding: 2, resource: this.pointerBuffer}
      ]
    });

    this.bindGroup1 = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(1),
      entries: [{
        binding: 0,
        resource: { buffer: contextUniform.uniformBuffer }
      }]
    });
  }

  update() {
    const commandEncoder = device.createCommandEncoder();
    const computePass = commandEncoder.beginComputePass();
      computePass.setPipeline(this.pipeline);
      computePass.setBindGroup(0, this.bindGroup0);
      computePass.setBindGroup(1, this.bindGroup1);
      computePass.dispatchWorkgroups(
        Math.ceil(this.gridSize / 4),
        Math.ceil(this.gridSize / 4),
        Math.ceil(this.gridSize / 4)
      );
      computePass.end();


      const readbackBuffer = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      commandEncoder.copyBufferToBuffer(this.pointerBuffer, 0, readbackBuffer, 0, 4);

      device.queue.submit([commandEncoder.finish()]);

      readbackBuffer.mapAsync(GPUMapMode.READ).then(() => {
        const result = new Uint32Array(readbackBuffer.getMappedRange());
        const pointer = result[0];
        console.log("Pointer:", pointer)
        readbackBuffer.unmap();
        device.queue.writeBuffer(this.pointerBuffer, 0, new Uint32Array([0]));

        const chunkBuffer = device.createBuffer({
          label: `Compact Octree`,
          size: pointer * 4,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        const commandEncoder = device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(this.octreeBuffer, 0, chunkBuffer, 0, pointer * 4);
        device.queue.submit([commandEncoder.finish()]);

        chunkBuffer.mapAsync(GPUMapMode.READ).then(() => {
          const result = new Uint32Array(chunkBuffer.getMappedRange());
          console.log("Octree Data:", result);
          chunkBuffer.unmap();
        });
        
      });
  }

  get worstCaseMaxNodes(): number {
    return Math.floor((Math.pow(8, this.maxDepth + 1) -1)/ 7);
  }

  get gridSize(): number {
    return Math.pow(2, this.maxDepth);
  }
}