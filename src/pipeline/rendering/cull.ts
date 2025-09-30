import {compression, contextUniform, device, gridSize} from "../../index";
import shader from "./cull.wgsl" with {type: "text"};
import {RenderTimer} from "../timing";
import {Chunk} from "../../chunk/chunk";

export class Cull {
	pipeline: GPUComputePipeline;

	// Per-chunk buffers
	chunkCounters = new Map<Chunk, GPUBuffer>();
	chunkCounterReadbacks = new Map<Chunk, GPUBuffer>();
	chunkIndicesBuffers = new Map<Chunk, GPUBuffer>();
	chunkIndicesReadbacks = new Map<Chunk, GPUBuffer>();

	chunkBindGroups = new Map<Chunk, GPUBindGroup>();
	chunkContextBindGroups = new Map<Chunk, GPUBindGroup>();
	chunkWorldPosBuffers = new Map<Chunk, GPUBuffer>();

	// output
	count: number = 0;
	timer: RenderTimer;
	private readbackInProgress = new Map<Chunk, boolean>();
	private framesSinceUpdate = new Map<Chunk, number>();
	private activeReadbackPromises = new Map<Chunk, Promise<void>>();

	constructor() {
		this.timer = new RenderTimer("cull");

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
	}

	get renderTime(): number {
		return this.timer.renderTime;
	}

	update(encoder: GPUCommandEncoder, chunk: Chunk) {
		const counter = this.chunkCounters.get(chunk);
		if (!counter) return;

		// Reset counter before culling
		device.queue.writeBuffer(counter, 0, new Uint32Array([0]));

		const pass = encoder.beginComputePass({
			timestampWrites: this.timer.getTimestampWrites(),
		});
		pass.setPipeline(this.pipeline);
		pass.setBindGroup(0, this.chunkBindGroups.get(chunk));
		pass.setBindGroup(1, this.chunkContextBindGroups.get(chunk));

		// Dispatch with 4x4x4 workgroup size
		const workgroupsPerDim = Math.ceil(gridSize / compression / 4);
		pass.dispatchWorkgroups(
			workgroupsPerDim,
			workgroupsPerDim,
			workgroupsPerDim,
		);
		pass.end();

		this.timer.resolveTimestamps(encoder);

		// Start async readback if not already in progress
		const frames = (this.framesSinceUpdate.get(chunk) ?? 0) + 1;
		this.framesSinceUpdate.set(chunk, frames);
		const inProgress = this.readbackInProgress.get(chunk) ?? false;
		if (!inProgress && frames >= 2) {
			// Update every 2 frames
			this.startAsyncReadback(chunk);
			this.framesSinceUpdate.set(chunk, 0);
		}
	}


	afterUpdate() {
		this.timer.readTimestamps();
	}

	registerChunk(chunk: Chunk) {
		const chunkLabel = `Chunk[${chunk.id}](${chunk.position[0]},${chunk.position[1]},${chunk.position[2]})`;

		// Create per-chunk counter buffer
		const counter = device.createBuffer({
			label: `${chunkLabel} Cull Counter`,
			size: 4,
			usage:
				GPUBufferUsage.STORAGE |
				GPUBufferUsage.COPY_SRC |
				GPUBufferUsage.COPY_DST,
		});

		const counterReadback = device.createBuffer({
			label: `${chunkLabel} Cull Counter Readback`,
			size: 4,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
		});

		// Create per-chunk indices buffer
		const indicesBuffer = device.createBuffer({
			label: `${chunkLabel} Cull Indices`,
			size: Math.pow(gridSize / compression, 3) * 4,
			usage:
				GPUBufferUsage.STORAGE |
				GPUBufferUsage.COPY_SRC |
				GPUBufferUsage.COPY_DST,
		});

		const indicesReadback = device.createBuffer({
			label: `${chunkLabel} Cull Indices Readback`,
			size: Math.pow(gridSize / compression, 3) * 4,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
		});

		device.queue.writeBuffer(counter, 0, new Uint32Array([0]));

		// Store per-chunk buffers
		this.chunkCounters.set(chunk, counter);
		this.chunkCounterReadbacks.set(chunk, counterReadback);
		this.chunkIndicesBuffers.set(chunk, indicesBuffer);
		this.chunkIndicesReadbacks.set(chunk, indicesReadback);

		// Create bind group with per-chunk buffers
		const bindGroup = device.createBindGroup({
			layout: this.pipeline.getBindGroupLayout(0),
			label: `${chunkLabel} Cull Bind Group`,
			entries: [
				{
					binding: 0,
					resource: counter,
				},
				{
					binding: 1,
					resource: {buffer: chunk.vertexCounts},
				},
				{
					binding: 2,
					resource: indicesBuffer,
				},
				{
					binding: 3,
					resource: {buffer: chunk.density},
				},
			],
		});

		// Create chunk world position buffer
		const chunkWorldPosBuffer = device.createBuffer({
			label: `${chunkLabel} Cull World Position`,
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

		// Create context bind group with chunk world position
		const contextBindGroup = device.createBindGroup({
			label: `${chunkLabel} Cull Context Bind Group`,
			layout: this.pipeline.getBindGroupLayout(1),
			entries: [
				{
					binding: 0,
					resource: {buffer: contextUniform.uniformBuffer},
				},
				{
					binding: 1,
					resource: {buffer: chunkWorldPosBuffer},
				},
			],
		});

		this.chunkBindGroups.set(chunk, bindGroup);
		this.chunkContextBindGroups.set(chunk, contextBindGroup);
		this.chunkWorldPosBuffers.set(chunk, chunkWorldPosBuffer);

		// Initialize per-chunk state
		this.readbackInProgress.set(chunk, false);
		this.framesSinceUpdate.set(chunk, 0);
	}

	async unregisterChunk(chunk: Chunk) {
		// Wait for any pending readback to complete before destroying buffers
		const pendingReadback = this.activeReadbackPromises.get(chunk);
		if (pendingReadback) {
			try {
				await pendingReadback;
			} catch {
				// Ignore errors, we're cleaning up anyway
			}
		}

		// Destroy per-chunk buffers
		const counter = this.chunkCounters.get(chunk);
		const counterReadback = this.chunkCounterReadbacks.get(chunk);
		const indicesBuffer = this.chunkIndicesBuffers.get(chunk);
		const indicesReadback = this.chunkIndicesReadbacks.get(chunk);
		const worldPosBuffer = this.chunkWorldPosBuffers.get(chunk);

		counter?.destroy();
		counterReadback?.destroy();
		indicesBuffer?.destroy();
		indicesReadback?.destroy();
		worldPosBuffer?.destroy();

		// Clean up maps
		this.chunkCounters.delete(chunk);
		this.chunkCounterReadbacks.delete(chunk);
		this.chunkIndicesBuffers.delete(chunk);
		this.chunkIndicesReadbacks.delete(chunk);
		this.chunkBindGroups.delete(chunk);
		this.chunkContextBindGroups.delete(chunk);
		this.chunkWorldPosBuffers.delete(chunk);
		this.readbackInProgress.delete(chunk);
		this.framesSinceUpdate.delete(chunk);
		this.activeReadbackPromises.delete(chunk);
	}

	private startAsyncReadback(chunk: Chunk) {
		const inProgress = this.readbackInProgress.get(chunk) ?? false;
		if (inProgress) return;

		this.readbackInProgress.set(chunk, true);

		// Start async readback without blocking
		const promise = this.performAsyncReadback(chunk)
			.then(() => {
				this.readbackInProgress.set(chunk, false);
				this.activeReadbackPromises.delete(chunk);
			})
			.catch((error) => {
				// Only log if it's not an abort error from chunk cleanup
				if (error.name !== 'AbortError') {
					console.warn("Culling readback failed:", error);
				}
				this.readbackInProgress.set(chunk, false);
				this.activeReadbackPromises.delete(chunk);
			});

		this.activeReadbackPromises.set(chunk, promise);
	}

	private async performAsyncReadback(chunk: Chunk): Promise<void> {
		const counter = this.chunkCounters.get(chunk);
		const counterReadback = this.chunkCounterReadbacks.get(chunk);
		const indicesBuffer = this.chunkIndicesBuffers.get(chunk);
		const indicesReadback = this.chunkIndicesReadbacks.get(chunk);

		if (!counter || !counterReadback || !indicesBuffer || !indicesReadback) {
			return;
		}

		// Wait a frame to ensure GPU work is submitted
		await new Promise((resolve) => requestAnimationFrame(resolve));

		// Check if chunk was cleaned up while we were waiting
		if (!this.chunkCounters.has(chunk)) {
			return; // Chunk was unregistered, abort readback
		}

		// Atomically copy both counter and indices in the same submission
		const encoder = device.createCommandEncoder();
		encoder.copyBufferToBuffer(
			counter,
			0,
			counterReadback,
			0,
			counter.size,
		);
		encoder.copyBufferToBuffer(
			indicesBuffer,
			0,
			indicesReadback,
			0,
			indicesBuffer.size,
		);
		device.queue.submit([encoder.finish()]);

		// Read counter first
		await counterReadback.mapAsync(GPUMapMode.READ);
		const counterData = counterReadback.getMappedRange();
		const newCount = new Uint32Array(counterData)[0];
		counterReadback.unmap();

		// Read indices using the count that was valid when buffers were copied
		await indicesReadback.mapAsync(GPUMapMode.READ);
		const indicesData = indicesReadback.getMappedRange();
		const readSize = Math.min(newCount * 4, indicesBuffer.size);
		chunk.indices = new Uint32Array(indicesData.slice(0, readSize));
		indicesReadback.unmap();

		// Update count last, after we've used the consistent value
		this.count = newCount;
	}
}
