import {compression, contextUniform, device, gridSize} from '../../index';
import shader from './mesh.wgsl';
import {RenderTimer} from '../timing';
import {EDGE_TABLE, TRIANGLE_TABLE} from './marchingCubeTables';
import {Chunk} from "../../chunk/chunk";

export class Mesh {
	pipeline: GPUComputePipeline;
	offsetBuffer: GPUBuffer;
	edgeTableBuffer: GPUBuffer;
	triangleTableBuffer: GPUBuffer;
	timer: RenderTimer;

	// Staging buffers for mesh generation (reused across all chunks)
	stagingVertices: GPUBuffer;
	stagingNormals: GPUBuffer;
	stagingMaterialColors: GPUBuffer;
	vertexCounter: GPUBuffer;
	vertexCounterReadback: GPUBuffer;
	maxVertices: number;

	chunkBindGroups = new Map<Chunk, GPUBindGroup>();
	chunkContextBindGroups = new Map<Chunk, GPUBindGroup>();
	chunkWorldPosBuffers = new Map<Chunk, GPUBuffer>();

	constructor() {
		this.timer = new RenderTimer('mesh');

		// Calculate max vertices for staging buffers
		const sSize = gridSize / compression;
		const sSize3 = sSize * sSize * sSize;
		this.maxVertices = sSize3 * 1536;

		this.offsetBuffer = device.createBuffer({
			label: 'Mesh Offset Buffer',
			size: 16, // vec3<u32> + padding = 16 bytes
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		// Create staging buffers for mesh generation (reused across all chunks)
		this.stagingVertices = device.createBuffer({
			label: 'Mesh Staging Vertices',
			size: 8 * this.maxVertices, // vec4<f16> = 8 bytes each
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
		});

		this.stagingNormals = device.createBuffer({
			label: 'Mesh Staging Normals',
			size: 8 * this.maxVertices, // vec3<f16> = 8 bytes each (with padding)
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
		});

		this.stagingMaterialColors = device.createBuffer({
			label: 'Mesh Staging Material Colors',
			size: 4 * this.maxVertices, // u32 = 4 bytes each
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
		});

		// Atomic counter for vertex allocation
		this.vertexCounter = device.createBuffer({
			label: 'Mesh Vertex Counter',
			size: 4, // atomic<u32> = 4 bytes
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
		});

		// Readback buffer for vertex counter
		this.vertexCounterReadback = device.createBuffer({
			label: 'Mesh Vertex Counter Readback',
			size: 4,
			usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
		});

		// Create and initialize lookup table buffers
		this.edgeTableBuffer = device.createBuffer({
			label: 'Mesh Edge Table',
			size: EDGE_TABLE.byteLength,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});
		device.queue.writeBuffer(this.edgeTableBuffer, 0, EDGE_TABLE);

		this.triangleTableBuffer = device.createBuffer({
			label: 'Mesh Triangle Table',
			size: TRIANGLE_TABLE.byteLength,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});
		device.queue.writeBuffer(this.triangleTableBuffer, 0, TRIANGLE_TABLE);

		const shaderModule = device.createShaderModule({
			code: shader,
		});

		this.pipeline = device.createComputePipeline({
			label: 'Mesh',
			layout: 'auto',
			compute: {
				module: shaderModule,
				entryPoint: 'main',
			},
		});


	}

	get renderTime(): number {
		return this.timer.renderTime;
	}

	update(
		encoder: GPUCommandEncoder,
		chunk: Chunk,
		bounds?: { min: [number, number, number]; max: [number, number, number] },
		resetCounter: boolean = true
	) {
		// Reset vertex counter only when starting a new mesh generation
		if (resetCounter) {
			encoder.clearBuffer(this.vertexCounter, 0, 4);
		}

		if (bounds) {
			// Calculate mesh chunk bounds in compressed grid space
			const sSize = gridSize / compression;
			const minChunk = [
				Math.max(0, Math.floor(bounds.min[0] / compression)),
				Math.max(0, Math.floor(bounds.min[1] / compression)),
				Math.max(0, Math.floor(bounds.min[2] / compression)),
			];
			const maxChunk = [
				Math.min(sSize - 1, Math.ceil(bounds.max[0] / compression)),
				Math.min(sSize - 1, Math.ceil(bounds.max[1] / compression)),
				Math.min(sSize - 1, Math.ceil(bounds.max[2] / compression)),
			];

			// Write offset to shader uniform
			const offsetData = new Uint32Array([
				minChunk[0],
				minChunk[1],
				minChunk[2],
				0,
			]); // padding for vec3<u32>
			device.queue.writeBuffer(this.offsetBuffer, 0, offsetData);

			const pass = encoder.beginComputePass({
				timestampWrites: this.timer.getTimestampWrites(),
			});
			pass.setPipeline(this.pipeline);
			pass.setBindGroup(0, this.chunkBindGroups.get(chunk));
			pass.setBindGroup(1, this.chunkContextBindGroups.get(chunk));

			// Dispatch only the affected chunks with 4x4x4 workgroup size
			const workgroupsX = Math.ceil((maxChunk[0] - minChunk[0] + 1) / 4);
			const workgroupsY = Math.ceil((maxChunk[1] - minChunk[1] + 1) / 4);
			const workgroupsZ = Math.ceil((maxChunk[2] - minChunk[2] + 1) / 4);

			pass.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);
			pass.end();
		} else {
			// Write zero offset for full grid update
			const offsetData = new Uint32Array([0, 0, 0, 0]);
			device.queue.writeBuffer(this.offsetBuffer, 0, offsetData);

			const pass = encoder.beginComputePass({
				timestampWrites: this.timer.getTimestampWrites(),
			});
			pass.setPipeline(this.pipeline);
			pass.setBindGroup(0, this.chunkBindGroups.get(chunk));
			pass.setBindGroup(1, this.chunkContextBindGroups.get(chunk));

			// Dispatch with 4x4x4 workgroup size for full grid
			const workgroupsPerDim = Math.ceil(gridSize / compression / 4);
			pass.dispatchWorkgroups(
				workgroupsPerDim,
				workgroupsPerDim,
				workgroupsPerDim
			);
			pass.end();
		}

		this.timer.resolveTimestamps(encoder);

		// Copy vertex counter for readback
		encoder.copyBufferToBuffer(
			this.vertexCounter,
			0,
			this.vertexCounterReadback,
			0,
			4
		);

		// Defer the efficient copy to after command encoder submission
		// We'll read the actual vertex count and copy only what's needed
		this.pendingChunkCopy = { chunk, encoder };
	}

	startMeshFinalization(): void {
		if (!this.pendingChunkCopy) return;

		// Start the async readback without awaiting
		this.pendingReadback = this.vertexCounterReadback.mapAsync(GPUMapMode.READ);
		this.readbackComplete = false;

		// Track completion without blocking
		this.pendingReadback.then(() => {
			this.readbackComplete = true;
		});
	}

	isMeshFinalizationReady(): boolean {
		return this.readbackComplete;
	}

	async completeMeshFinalization(): Promise<{ chunk: Chunk; buffersResized: boolean } | null> {
		if (!this.pendingChunkCopy || !this.pendingReadback) return null;

		const { chunk } = this.pendingChunkCopy;
		this.pendingChunkCopy = null;

		// Wait for the readback that was started earlier
		await this.pendingReadback;
		this.pendingReadback = null;
		this.readbackComplete = false;
		const counterData = new Uint32Array(this.vertexCounterReadback.getMappedRange());
		const actualVertexCount = counterData[0];
		this.vertexCounterReadback.unmap();

		if (actualVertexCount === 0) {
			console.warn(`Chunk ${chunk.id}: No vertices generated!`);
			return { chunk, buffersResized: false };
		}

		if (actualVertexCount > this.maxVertices) {
			console.error(`CRITICAL: Vertex count ${actualVertexCount} exceeds staging buffer capacity ${this.maxVertices}!`);
			console.error(`This means vertices were written out of bounds and data is corrupted!`);
		}

		// Ensure chunk buffers are sized appropriately (with some headroom)
		const targetSize = Math.ceil(actualVertexCount * 1.2);
		const buffersResized = this.ensureChunkBufferSize(chunk, targetSize);

		// Create new encoder for the copy operation
		const copyEncoder = device.createCommandEncoder({
			label: 'Mesh Copy'
		});

		// Copy only the used portion of staging buffers
		const verticesBytes = actualVertexCount * 8;
		const normalsBytes = actualVertexCount * 8;
		const colorsBytes = actualVertexCount * 4;

		// Ensure we don't exceed buffer sizes
		if (verticesBytes > chunk.vertices.size) {
			console.error(`Vertices overflow! Need ${verticesBytes}, have ${chunk.vertices.size}`);
		}
		if (normalsBytes > chunk.normals.size) {
			console.error(`Normals overflow! Need ${normalsBytes}, have ${chunk.normals.size}`);
		}
		if (colorsBytes > chunk.colors.size) {
			console.error(`Colors overflow! Need ${colorsBytes}, have ${chunk.colors.size}`);
		}

		copyEncoder.copyBufferToBuffer(
			this.stagingVertices,
			0,
			chunk.vertices,
			0,
			Math.min(verticesBytes, chunk.vertices.size)
		);

		copyEncoder.copyBufferToBuffer(
			this.stagingNormals,
			0,
			chunk.normals,
			0,
			Math.min(normalsBytes, chunk.normals.size)
		);

		copyEncoder.copyBufferToBuffer(
			this.stagingMaterialColors,
			0,
			chunk.materialColors,
			0,
			Math.min(colorsBytes, chunk.materialColors.size)
		);

		copyEncoder.copyBufferToBuffer(
			this.stagingMaterialColors,
			0,
			chunk.colors,
			0,
			Math.min(colorsBytes, chunk.colors.size)
		);

		device.queue.submit([copyEncoder.finish()]);

		return { chunk, buffersResized };
	}

	private pendingChunkCopy: { chunk: Chunk; encoder: GPUCommandEncoder } | null = null;
	private pendingReadback: Promise<void> | null = null;
	private readbackComplete: boolean = false;

	afterUpdate() {
		this.timer.readTimestamps();
	}

	ensureChunkBufferSize(chunk: Chunk, vertexCount: number): boolean {
		const requiredVerticesSize = vertexCount * 8;
		const requiredNormalsSize = vertexCount * 8;
		const requiredColorsSize = vertexCount * 4;

		// Check if buffers need resizing (grow by 1.5x to avoid frequent resizes)
		const needsResize =
			chunk.vertices.size < requiredVerticesSize ||
			chunk.normals.size < requiredNormalsSize ||
			chunk.materialColors.size < requiredColorsSize ||
			chunk.colors.size < requiredColorsSize;

		if (needsResize) {
			const newVertexCount = Math.ceil(vertexCount * 1.5);
			const chunkLabel = `Chunk[${chunk.id}](${chunk.position[0]},${chunk.position[1]},${chunk.position[2]})`;

			// Destroy old buffers
			chunk.vertices.destroy();
			chunk.normals.destroy();
			chunk.materialColors.destroy();
			chunk.colors.destroy();

			// Create new larger buffers
			chunk.vertices = device.createBuffer({
				label: `${chunkLabel} Vertices`,
				size: newVertexCount * 8,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
			});

			chunk.normals = device.createBuffer({
				label: `${chunkLabel} Normals`,
				size: newVertexCount * 8,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
			});

			chunk.materialColors = device.createBuffer({
				label: `${chunkLabel} Material Colors`,
				size: newVertexCount * 4,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
			});

			chunk.colors = device.createBuffer({
				label: `${chunkLabel} Lit Colors`,
				size: newVertexCount * 4,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
			});

			return true; // Buffers were resized
		}
		return false; // No resize needed
	}

	registerChunk(chunk: Chunk) {

		const bindGroup = device.createBindGroup({
			label: 'Mesh',
			layout: this.pipeline.getBindGroupLayout(0),
			entries: [
				{
					binding: 0,
					resource: {buffer: chunk.voxelData}, // Input
				},
				{
					binding: 1,
					resource: {buffer: chunk.vertexCounts}, // Output
				},
				{
					binding: 2,
					resource: {buffer: this.stagingVertices}, // Output (staging)
				},
				{
					binding: 3,
					resource: {buffer: this.stagingNormals}, // Output (staging)
				},
				{
					binding: 4,
					resource: {buffer: this.stagingMaterialColors}, // Output (staging)
				},
				{
					binding: 5,
					resource: {buffer: chunk.commands}, // Output
				},
				{
					binding: 6,
					resource: {buffer: chunk.density}, // Output
				},
				{
					binding: 7,
					resource: {buffer: this.vertexCounter}, // Atomic counter
				},
			],
		});

		// Create chunk world position buffer
		const chunkLabel = `Chunk[${chunk.id}](${chunk.position[0]},${chunk.position[1]},${chunk.position[2]})`;
		const chunkWorldPosBuffer = device.createBuffer({
			label: `${chunkLabel} Mesh World Position`,
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
			label: 'Mesh Context',
			layout: this.pipeline.getBindGroupLayout(1),
			entries: [
				{
					binding: 0,
					resource: {buffer: contextUniform.uniformBuffer},
				},
				{
					binding: 1,
					resource: {buffer: this.offsetBuffer},
				},
				{
					binding: 2,
					resource: {buffer: this.edgeTableBuffer},
				},
				{
					binding: 3,
					resource: {buffer: this.triangleTableBuffer},
				},
				{
					binding: 4,
					resource: {buffer: chunkWorldPosBuffer},
				},
			],
		});

		this.chunkBindGroups.set(chunk, bindGroup);
		this.chunkContextBindGroups.set(chunk, contextBindGroup);
		this.chunkWorldPosBuffers.set(chunk, chunkWorldPosBuffer);
	}

	unregisterChunk(chunk: Chunk) {
		const worldPosBuffer = this.chunkWorldPosBuffers.get(chunk);
		if (worldPosBuffer) {
			worldPosBuffer.destroy();
		}
		this.chunkBindGroups.delete(chunk);
		this.chunkContextBindGroups.delete(chunk);
		this.chunkWorldPosBuffers.delete(chunk);
	}
}
