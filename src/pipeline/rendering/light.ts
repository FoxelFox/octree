import {compression, contextUniform, device, gridSize} from "../../index";
import shader from "./light.wgsl" with {type: "text"};
import vertexLightShader from "./vertex_light.wgsl" with {type: "text"};
import {RenderTimer} from "../timing";
import {Chunk} from "../../chunk/chunk";

export class Light {

	// Compute pipeline for light propagation
	pipeline: GPUComputePipeline;
	vertexLightPipeline: GPUComputePipeline;
	bindGroups = new Map<Chunk, GPUBindGroup>();
	vertexLightBindGroups = new Map<Chunk, GPUBindGroup>();
	chunkContextBindGroups = new Map<Chunk, GPUBindGroup>();
	vertexLightContextBindGroups = new Map<Chunk, GPUBindGroup>();
	chunkWorldPosBuffers = new Map<Chunk, GPUBuffer>();
	vertexLightCombinedBuffers = new Map<Chunk, GPUBuffer>();
	// Configuration uniforms
	configBuffer: GPUBuffer;
	// Timer for profiling
	timer: RenderTimer;
	// Per-chunk simulation state
	private chunkIterationCounts = new Map<Chunk, number>();
	private chunkNeedsUpdate = new Map<Chunk, boolean>();
	private chunkNeedsRebake = new Map<Chunk, boolean>();
	private maxIterations = 16; // Number of iterations to propagate light
	private getNeighborChunks?: (chunk: Chunk) => Chunk[];
	private readonly lightSegmentSize: number;

	constructor() {
		const sSize = gridSize / compression;
		const cellCount = sSize * sSize * sSize;
		this.lightSegmentSize = cellCount * 8;

		this.timer = new RenderTimer("light");
		this.initBuffers();

		// Create compute pipeline
		this.pipeline = device.createComputePipeline({
			label: "Light Floodfill",
			layout: "auto",
			compute: {
				module: device.createShaderModule({
					code: shader,
				}),
				entryPoint: "main",
			},
		});

		// Create vertex light baking pipeline
		this.vertexLightPipeline = device.createComputePipeline({
			label: "Vertex Light Baking",
			layout: "auto",
			compute: {
				module: device.createShaderModule({
					code: vertexLightShader,
				}),
				entryPoint: "main",
			},
		});
	}

	get renderTime(): number {
		return this.timer.renderTime;
	}

	update(encoder: GPUCommandEncoder, chunk: Chunk, getNeighborChunks?: (chunk: Chunk) => Chunk[]) {
		const needsUpdate = this.chunkNeedsUpdate.get(chunk) ?? false;
		if (!needsUpdate) return;

		// Store neighbor chunk getter for use in unregisterChunk
		if (getNeighborChunks) {
			this.getNeighborChunks = getNeighborChunks;
		}

		// Recreate bind group with current neighbor data
		if (getNeighborChunks) {
			this.bindGroups.set(chunk, this.createComputeBindGroup(chunk, getNeighborChunks));
			this.vertexLightBindGroups.set(chunk, this.createVertexLightBindGroup(chunk));
		}

		const bindGroup = this.bindGroups.get(chunk);
		if (!bindGroup) return;

		// Run single pass of light propagation
		const pass = encoder.beginComputePass({
			label: `Light Propagation`,
			timestampWrites: this.timer.getTimestampWrites(),
		});

		pass.setPipeline(this.pipeline);
		pass.setBindGroup(0, bindGroup);
		pass.setBindGroup(1, this.chunkContextBindGroups.get(chunk));

		// Dispatch with 4x4x4 workgroup size to match mesh generation
		const sSize = gridSize / compression;
		const workgroupsPerDim = Math.ceil(sSize / 4);
		pass.dispatchWorkgroups(
			workgroupsPerDim,
			workgroupsPerDim,
			workgroupsPerDim,
		);

		pass.end();

		if (this.timer.getTimestampWrites()) {
			this.timer.resolveTimestamps(encoder);
		}

		const iterationCount = (this.chunkIterationCounts.get(chunk) ?? 0) + 1;
		this.chunkIterationCounts.set(chunk, iterationCount);

		// Bake lighting into vertex colors when light stabilizes or when forced
		const needsRebake = this.chunkNeedsRebake.get(chunk) ?? false;
		if (iterationCount >= this.maxIterations * 2 || needsRebake) {
			this.chunkNeedsUpdate.set(chunk, false);
			this.chunkNeedsRebake.set(chunk, false);
			this.bakeVertexLighting(encoder, chunk, getNeighborChunks);
		}
	}

	bakeVertexLighting(encoder: GPUCommandEncoder, chunk: Chunk, getNeighborChunks?: (chunk: Chunk) => Chunk[]) {
		const vertexLightBindGroup = this.vertexLightBindGroups.get(chunk);
		if (!vertexLightBindGroup) return;

		this.populateVertexLightCombinedBuffer(encoder, chunk, getNeighborChunks);

		const pass = encoder.beginComputePass({
			label: `Vertex Light Baking`,
		});

		// Create light bind group for vertex shader
		const lightBindGroup = this.createVertexLightDataBindGroup(chunk);

		pass.setPipeline(this.vertexLightPipeline);
		pass.setBindGroup(0, vertexLightBindGroup);
		pass.setBindGroup(1, lightBindGroup);
		pass.setBindGroup(2, this.vertexLightContextBindGroups.get(chunk)!);

		// Dispatch for all potential vertices (shader will skip empty meshlets via vertexCounts)
		const sSize = gridSize / compression;
		const meshletCount = sSize * sSize * sSize;
		const verticesPerMeshlet = 1536;
		const totalVertices = meshletCount * verticesPerMeshlet;
		const workgroups = Math.ceil(totalVertices / 64);

		// Split across X and Y dimensions to stay within limits (max 65535 per dimension)
		const maxWorkgroupsPerDim = 65535;
		const workgroupsX = Math.min(workgroups, maxWorkgroupsPerDim);
		const workgroupsY = Math.ceil(workgroups / maxWorkgroupsPerDim);
		pass.dispatchWorkgroups(workgroupsX, workgroupsY);

		pass.end();
	}

	// Force a light update (call when voxels are modified)
	invalidate(chunk?: Chunk) {
		if (chunk) {
			this.chunkNeedsUpdate.set(chunk, true);
			this.chunkIterationCounts.set(chunk, 0);
			// Force rebake on next frame to avoid showing stale lighting
			this.chunkNeedsRebake.set(chunk, true);
		} else {
			// Invalidate all chunks
			for (const c of this.chunkNeedsUpdate.keys()) {
				this.chunkNeedsUpdate.set(c, true);
				this.chunkIterationCounts.set(c, 0);
				this.chunkNeedsRebake.set(c, true);
			}
		}
	}

	afterUpdate() {
		this.timer.readTimestamps();
	}

	// Dummy buffer for missing neighbors
	private dummyLightBuffer: GPUBuffer | null = null;

	private getDummyLightBuffer(): GPUBuffer {
		if (!this.dummyLightBuffer) {
			const sSize = gridSize / compression;
			const totalCells = sSize * sSize * sSize;
			const dummyData = new Float32Array(totalCells * 2);
			// Fill with "no light" values
			for (let i = 0; i < totalCells; i++) {
				dummyData[i * 2] = 0.0; // no light
				dummyData[i * 2 + 1] = 1.0; // full shadow
			}
			this.dummyLightBuffer = device.createBuffer({
				label: 'Light Dummy Buffer',
				size: dummyData.byteLength,
				usage:
					GPUBufferUsage.STORAGE |
					GPUBufferUsage.COPY_DST |
					GPUBufferUsage.COPY_SRC,
			});
			device.queue.writeBuffer(this.dummyLightBuffer, 0, dummyData);
		}
		return this.dummyLightBuffer;
	}

	registerChunk(chunk: Chunk) {
		// Initialize light buffer with skylight
		this.initializeLighting(chunk);

		// Initial bind group without neighbors (will be updated on first light update)
		this.bindGroups.set(chunk, this.createComputeBindGroup(chunk));
		this.vertexLightBindGroups.set(chunk, this.createVertexLightBindGroup(chunk));

		// Mark chunk as needing light updates
		this.chunkNeedsUpdate.set(chunk, true);
		this.chunkIterationCounts.set(chunk, 0);

		// Create chunk world position buffer
		const chunkLabel = `Chunk[${chunk.id}](${chunk.position[0]},${chunk.position[1]},${chunk.position[2]})`;
		const chunkWorldPosBuffer = device.createBuffer({
			label: `${chunkLabel} Light World Position`,
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
			label: "Light Context",
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

		this.chunkContextBindGroups.set(chunk, contextBindGroup);
		this.chunkWorldPosBuffers.set(chunk, chunkWorldPosBuffer);

		const combinedLightBuffer = device.createBuffer({
			label: `${chunkLabel} Vertex Light Combined`,
			size: this.lightSegmentSize * 7,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		});
		this.vertexLightCombinedBuffers.set(chunk, combinedLightBuffer);

		// Create vertex light context bind group
		const vertexLightContextBindGroup = device.createBindGroup({
			label: "Vertex Light Context",
			layout: this.vertexLightPipeline.getBindGroupLayout(2),
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

		this.vertexLightContextBindGroups.set(chunk, vertexLightContextBindGroup);
	}

	unregisterChunk(chunk: Chunk) {
		// Invalidate and recreate bind groups in chunks that reference this chunk as a neighbor
		if (this.getNeighborChunks) {
			for (const otherChunk of this.bindGroups.keys()) {
				const neighbors = this.getNeighborChunks(otherChunk);
				if (neighbors.includes(chunk)) {
					// Recreate bind group with dummy buffer for destroyed neighbor
					this.bindGroups.set(otherChunk, this.createComputeBindGroup(otherChunk, this.getNeighborChunks));
					this.vertexLightBindGroups.set(otherChunk, this.createVertexLightBindGroup(otherChunk));
				}
			}
		}

		const worldPosBuffer = this.chunkWorldPosBuffers.get(chunk);
		if (worldPosBuffer) {
			worldPosBuffer.destroy();
		}
		this.bindGroups.delete(chunk);
		this.vertexLightBindGroups.delete(chunk);
		this.chunkContextBindGroups.delete(chunk);
		this.vertexLightContextBindGroups.delete(chunk);
		const combinedBuffer = this.vertexLightCombinedBuffers.get(chunk);
		combinedBuffer?.destroy();
		this.vertexLightCombinedBuffers.delete(chunk);
		this.chunkWorldPosBuffers.delete(chunk);
		this.chunkNeedsUpdate.delete(chunk);
		this.chunkIterationCounts.delete(chunk);
		this.chunkNeedsRebake.delete(chunk);
	}

	private initBuffers() {


		// Configuration buffer for simulation parameters
		this.configBuffer = device.createBuffer({
			label: 'Light Config Buffer',
			size: 32, // 8 * 4 bytes for configuration
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		// Initialize with default values
		this.updateConfig();
	}

	private updateConfig() {
		const config = new Float32Array(8);
		config[0] = this.maxIterations; // max_iterations
		config[1] = 0.85; // light_attenuation (how much light dims per step)
		config[2] = 0.95; // shadow_softness (how soft shadows are)
		config[3] = 1.0;  // skylight_intensity
		// config[4-7] reserved for future use

		device.queue.writeBuffer(this.configBuffer, 0, config);
	}

	private initializeLighting(chunk: Chunk) {
		// Initialize the light buffer with skylight from above
		const sSize = gridSize / compression;
		const totalCells = sSize * sSize * sSize;
		const initData = new Float32Array(totalCells * 2); // RG format

		// Calculate world Y position of this chunk
		const chunkWorldY = chunk.position[1] * gridSize;

		// Fill with initial skylight values - based on world position
		for (let z = 0; z < sSize; z++) {
			for (let y = 0; y < sSize; y++) {
				for (let x = 0; x < sSize; x++) {
					const index = (z * sSize * sSize + y * sSize + x) * 2;

					// Calculate world Y for this cell
					const worldY = chunkWorldY + y * compression;

					// Add skylight to top layer if in upper chunks (world Y > 0)
					const isTopLayer = (y >= sSize - 1);
					const isUpperChunk = worldY > 0;
					const skylight = (isTopLayer && isUpperChunk) ? Math.min(worldY / 256.0, 1.0) : 0.0;

					initData[index] = skylight;     // R: light intensity
					initData[index + 1] = isTopLayer ? 0.0 : 1.0;      // G: shadow factor
				}
			}
		}

		// Initialize light buffer
		device.queue.writeBuffer(chunk.light, 0, initData);
	}

	private createComputeBindGroup(chunk: Chunk, getNeighborChunks?: (chunk: Chunk) => Chunk[]): GPUBindGroup {
		const dummyBuffer = this.getDummyLightBuffer();

		// Find neighbor chunks in each direction
		let neighborNX: Chunk | undefined;
		let neighborPX: Chunk | undefined;
		let neighborNY: Chunk | undefined;
		let neighborPY: Chunk | undefined;
		let neighborNZ: Chunk | undefined;
		let neighborPZ: Chunk | undefined;

		if (getNeighborChunks) {
			const neighbors = getNeighborChunks(chunk);
			for (const neighbor of neighbors) {
				const dx = neighbor.position[0] - chunk.position[0];
				const dy = neighbor.position[1] - chunk.position[1];
				const dz = neighbor.position[2] - chunk.position[2];

				if (dx === -1 && dy === 0 && dz === 0) neighborNX = neighbor;
				else if (dx === 1 && dy === 0 && dz === 0) neighborPX = neighbor;
				else if (dx === 0 && dy === -1 && dz === 0) neighborNY = neighbor;
				else if (dx === 0 && dy === 1 && dz === 0) neighborPY = neighbor;
				else if (dx === 0 && dy === 0 && dz === -1) neighborNZ = neighbor;
				else if (dx === 0 && dy === 0 && dz === 1) neighborPZ = neighbor;
			}
		}

		return device.createBindGroup({
			label: `Light Data Chunk ${chunk.id}`,
			layout: this.pipeline.getBindGroupLayout(0),
			entries: [
				{
					binding: 0,
					resource: {buffer: chunk.density},
				},
				{
					binding: 1,
					resource: {buffer: chunk.light},
				},
				{
					binding: 2,
					resource: {buffer: this.configBuffer},
				},
				{
					binding: 3,
					resource: {buffer: neighborNX?.light ?? dummyBuffer},
				},
				{
					binding: 4,
					resource: {buffer: neighborPX?.light ?? dummyBuffer},
				},
				{
					binding: 5,
					resource: {buffer: neighborNY?.light ?? dummyBuffer},
				},
				{
					binding: 6,
					resource: {buffer: neighborPY?.light ?? dummyBuffer},
				},
				{
					binding: 7,
					resource: {buffer: neighborNZ?.light ?? dummyBuffer},
				},
				{
					binding: 8,
					resource: {buffer: neighborPZ?.light ?? dummyBuffer},
				},
			],
		});
	}

	private createVertexLightBindGroup(chunk: Chunk): GPUBindGroup {
		return device.createBindGroup({
			label: `Vertex Light Baking Chunk ${chunk.id}`,
			layout: this.vertexLightPipeline.getBindGroupLayout(0),
			entries: [
				{
					binding: 0,
					resource: {buffer: chunk.vertices},
				},
				{
					binding: 1,
					resource: {buffer: chunk.materialColors},  // Read original material colors
				},
				{
					binding: 2,
					resource: {buffer: chunk.normals},
				},
				{
					binding: 3,
					resource: {buffer: chunk.colors},  // Write lit colors
				},
				{
					binding: 4,
					resource: {buffer: chunk.vertexCounts},  // Vertex counts per meshlet
				},
			],
		});
	}

	private createVertexLightDataBindGroup(chunk: Chunk): GPUBindGroup {
		const combinedBuffer = this.vertexLightCombinedBuffers.get(chunk);
		if (!combinedBuffer) {
			throw new Error("Missing combined light buffer for chunk");
		}

		return device.createBindGroup({
			label: `Vertex Light Data Chunk ${chunk.id}`,
			layout: this.vertexLightPipeline.getBindGroupLayout(1),
			entries: [
				{ binding: 0, resource: { buffer: combinedBuffer } },
			],
		});
	}

	private populateVertexLightCombinedBuffer(
		encoder: GPUCommandEncoder,
		chunk: Chunk,
		getNeighborChunks?: (chunk: Chunk) => Chunk[],
	) {
		const combinedBuffer = this.vertexLightCombinedBuffers.get(chunk);
		if (!combinedBuffer) {
			return;
		}

		const neighborGetter = getNeighborChunks ?? this.getNeighborChunks;
		let neighborNX: Chunk | undefined;
		let neighborPX: Chunk | undefined;
		let neighborNY: Chunk | undefined;
		let neighborPY: Chunk | undefined;
		let neighborNZ: Chunk | undefined;
		let neighborPZ: Chunk | undefined;

		if (neighborGetter) {
			const neighbors = neighborGetter(chunk);
			for (const neighbor of neighbors) {
				const dx = neighbor.position[0] - chunk.position[0];
				const dy = neighbor.position[1] - chunk.position[1];
				const dz = neighbor.position[2] - chunk.position[2];

				if (dx === -1 && dy === 0 && dz === 0) neighborNX = neighbor;
				else if (dx === 1 && dy === 0 && dz === 0) neighborPX = neighbor;
				else if (dx === 0 && dy === -1 && dz === 0) neighborNY = neighbor;
				else if (dx === 0 && dy === 1 && dz === 0) neighborPY = neighbor;
				else if (dx === 0 && dy === 0 && dz === -1) neighborNZ = neighbor;
				else if (dx === 0 && dy === 0 && dz === 1) neighborPZ = neighbor;
			}
		}

		const sources = [
			chunk.light,
			neighborNX?.light ?? this.getDummyLightBuffer(),
			neighborPX?.light ?? this.getDummyLightBuffer(),
			neighborNY?.light ?? this.getDummyLightBuffer(),
			neighborPY?.light ?? this.getDummyLightBuffer(),
			neighborNZ?.light ?? this.getDummyLightBuffer(),
			neighborPZ?.light ?? this.getDummyLightBuffer(),
		];

		sources.forEach((source, index) => {
			encoder.copyBufferToBuffer(
				source,
				0,
				combinedBuffer,
				this.lightSegmentSize * index,
				this.lightSegmentSize,
			);
		});
	}
}
