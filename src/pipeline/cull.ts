import {contextUniform, device, gridSize} from "..";
import {Mesh} from "./mesh";
import {Noise} from "./noise";
import shader from "./cull.wgsl" with {type: "text"};

export class Cull {
	counter: GPUBuffer;
	counterReadback: GPUBuffer;
	bindGroup: GPUBindGroup;
	contextBindGroup: GPUBindGroup;
	pipeline: GPUComputePipeline;
	indicesBuffer: GPUBuffer;
	indicesReadback: GPUBuffer;

	// output
	count: number = 0;
	indices: Uint32Array = new Uint32Array(0);
	private readbackInProgress = false;
	private framesSinceUpdate = 0;

	init(noise: Noise, mesh: Mesh) {
		this.counter = device.createBuffer({
			size: 4,
			usage:
				GPUBufferUsage.STORAGE |
				GPUBufferUsage.COPY_SRC |
				GPUBufferUsage.COPY_DST,
		});

		this.counterReadback = device.createBuffer({
			size: this.counter.size,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
		});

		this.indicesBuffer = device.createBuffer({
			size: Math.pow(gridSize / 8, 3) * 4,
			usage:
				GPUBufferUsage.STORAGE |
				GPUBufferUsage.COPY_SRC |
				GPUBufferUsage.COPY_DST,
		});

		this.indicesReadback = device.createBuffer({
			size: this.indicesBuffer.size,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
		});

		device.queue.writeBuffer(this.counter, 0, new Uint32Array([0]));

		this.pipeline = device.createComputePipeline({
			layout: "auto",
			label: "Cull",
			compute: {
				module: device.createShaderModule({
					code: shader,
				}),
				entryPoint: "main",
			},
		});

		this.bindGroup = device.createBindGroup({
			layout: this.pipeline.getBindGroupLayout(0),
			label: "Cull",
			entries: [
				{
					binding: 0,
					resource: this.counter,
				},
				{
					binding: 1,
					resource: mesh.meshes,
				},
				{
					binding: 2,
					resource: this.indicesBuffer,
				},
			],
		});

		this.contextBindGroup = device.createBindGroup({
			label: "Cull Context",
			layout: this.pipeline.getBindGroupLayout(1),
			entries: [
				{
					binding: 0,
					resource: {buffer: contextUniform.uniformBuffer},
				},
			],
		});
	}

	update(encoder: GPUCommandEncoder) {
		// Reset counter before culling
		device.queue.writeBuffer(this.counter, 0, new Uint32Array([0]));

		const pass = encoder.beginComputePass();
		pass.setPipeline(this.pipeline);
		pass.setBindGroup(0, this.bindGroup);
		pass.setBindGroup(1, this.contextBindGroup);

		// Dispatch with 4x4x4 workgroup size
		const workgroupsPerDim = Math.ceil(gridSize / 8 / 4);
		pass.dispatchWorkgroups(
			workgroupsPerDim,
			workgroupsPerDim,
			workgroupsPerDim,
		);
		pass.end();

		// Start async readback if not already in progress
		this.framesSinceUpdate++;
		if (!this.readbackInProgress && this.framesSinceUpdate >= 2) { // Update every 2 frames
			this.startAsyncReadback();
			this.framesSinceUpdate = 0;
		}
	}

	private startAsyncReadback() {
		if (this.readbackInProgress) return;

		this.readbackInProgress = true;

		// Start async readback without blocking
		this.performAsyncReadback().then(() => {
			this.readbackInProgress = false;
		}).catch((error) => {
			console.warn("Culling readback failed:", error);
			this.readbackInProgress = false;
		});
	}

	private async performAsyncReadback(): Promise<void> {
		// Wait a frame to ensure GPU work is submitted
		await new Promise(resolve => requestAnimationFrame(resolve));

		// Atomically copy both counter and indices in the same submission
		const encoder = device.createCommandEncoder();
		encoder.copyBufferToBuffer(
			this.counter,
			0,
			this.counterReadback,
			0,
			this.counter.size,
		);
		encoder.copyBufferToBuffer(
			this.indicesBuffer,
			0,
			this.indicesReadback,
			0,
			this.indicesBuffer.size,
		);
		device.queue.submit([encoder.finish()]);

		// Read counter first
		await this.counterReadback.mapAsync(GPUMapMode.READ);
		const counterData = this.counterReadback.getMappedRange();
		const newCount = new Uint32Array(counterData)[0];
		this.counterReadback.unmap();

		// Read indices using the count that was valid when buffers were copied
		await this.indicesReadback.mapAsync(GPUMapMode.READ);
		const indicesData = this.indicesReadback.getMappedRange();
		const readSize = Math.min(newCount * 4, this.indicesBuffer.size);
		this.indices = new Uint32Array(indicesData.slice(0, readSize));
		this.indicesReadback.unmap();

		// Update count last, after we've used the consistent value
		this.count = newCount;
	}

	async readback(): Promise<void> {
		{
			// first read how many indices we have
			const encoder = device.createCommandEncoder();
			encoder.copyBufferToBuffer(
				this.counter,
				0,
				this.counterReadback,
				0,
				this.counter.size,
			);
			device.queue.submit([encoder.finish()]);

			await this.counterReadback.mapAsync(GPUMapMode.READ);
			const data = this.counterReadback.getMappedRange();
			this.count = new Uint32Array(data)[0];
			this.counterReadback.unmap();

			console.log("Index Count", this.count);
		}

		await device.queue.onSubmittedWorkDone();

		{
			// now only read the indices we need
			const encoder = device.createCommandEncoder();
			encoder.copyBufferToBuffer(
				this.indicesBuffer,
				0,
				this.indicesReadback,
				0,
				this.count * 4,
			);
			device.queue.submit([encoder.finish()]);

			await this.indicesReadback.mapAsync(GPUMapMode.READ);
			const data = this.indicesReadback.getMappedRange();
			this.indices = new Uint32Array(data.slice());
			this.indicesReadback.unmap();

			console.log("Indices", this.indices);
		}
	}
}
