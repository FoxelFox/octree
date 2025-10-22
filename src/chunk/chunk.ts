import {compression, device, gridSize} from "../index";

export class Chunk {
	id: number
	position: number[]
	voxelData: GPUBuffer;    // density (f32) + color (u32) = 8 bytes per voxel
	vertexCounts: GPUBuffer; // just for the cull pipeline to fast ignore empty meshlets
	commands: GPUBuffer;     // indirect draw commands

	vertices: GPUBuffer;
	normals: GPUBuffer;
	materialColors: GPUBuffer;  // original material colors from voxels
	colors: GPUBuffer;          // lit colors (materialColors * lighting)
	density: GPUBuffer;         // used for density occlusion culling and light blocker
	indices: Uint32Array;       // indices for meshlets that are actually needed to be rendered


	light: GPUBuffer;		// light data
	chunkLabel: string;


	constructor(id: number, position: number[]) {
		this.id = id;
		this.position = position;

		// Allocate 257³ voxels (gridSize + 1) to handle chunk borders
		// Each voxel: density (f32) + color (u32) = 8 bytes
		const voxelGridSize = gridSize + 1;
		const size = Math.pow(voxelGridSize, 3) * 8;
		const sSize = gridSize / compression;
		const sSize3 = sSize * sSize * sSize;

		this.chunkLabel = `Chunk[${id}](${position[0]},${position[1]},${position[2]})`;

		this.voxelData = device.createBuffer({
			label: `${this.chunkLabel} Voxel Data`,
			size,
			usage:
				GPUBufferUsage.STORAGE |
				GPUBufferUsage.COPY_DST |
				GPUBufferUsage.COPY_SRC,
		});

		// Separate buffers for mesh data
		this.vertexCounts = device.createBuffer({
			label: `${this.chunkLabel} Vertex Counts`,
			size: 4 * sSize3, // u32 = 4 bytes each
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
		});

		this.commands = device.createBuffer({
			label: `${this.chunkLabel} Commands`,
			size: 16 * sSize3,
			usage:
				GPUBufferUsage.STORAGE |
				GPUBufferUsage.COPY_SRC |
				GPUBufferUsage.COPY_DST |
				GPUBufferUsage.INDIRECT,
		});

		this.density = device.createBuffer({
			label: `${this.chunkLabel} Density`,
			size: 4 * sSize3,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
		});


		// Create light buffer: stores light intensity (R) and shadow factor (G) for each compressed cell
		// Format: RG32Float - R = light intensity, G = shadow factor (0.0 = fully lit, 1.0 = fully shadowed)
		this.light = device.createBuffer({
			label: `${this.chunkLabel} Light`,
			size: sSize3 * 8, // 2 * 4 bytes per cell (RG32Float)
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
		});
	}

	setVoxelData(data: Float32Array) {
		device.queue.writeBuffer(this.voxelData, 0, data);
	}

	setColors(data: Uint32Array) {

		if (this.colors) {
			this.colors.destroy();
		}

		this.colors = device.createBuffer({
			label: `${this.chunkLabel} Lit Colors`,
			size: data.byteLength,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
		});

		device.queue.writeBuffer(this.colors, 0, data);
	}

	setNormals(data: Float32Array) {

		if (this.normals) {
			this.normals.destroy();
		}

		this.normals = device.createBuffer({
			label: `${this.chunkLabel} Normals`,
			size: data.byteLength,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
		});

		device.queue.writeBuffer(this.normals, 0, data);
	}

	setMaterialColors(data: Uint32Array) {

		if (this.materialColors) {
			this.materialColors.destroy();
		}

		this.materialColors = device.createBuffer({
			label: `${this.chunkLabel} Material Colors`,
			size: data.byteLength,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
		});

		device.queue.writeBuffer(this.materialColors, 0, data);
	}

	setVertices(data: Float32Array) {

		if (this.vertices) {
			this.vertices.destroy();
		}

		this.vertices = device.createBuffer({
			label: `${this.chunkLabel} Vertices`,
			size: data.byteLength,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
		});

		device.queue.writeBuffer(this.vertices, 0, data);
	}

	setVertexCounts(data: Uint32Array) {
		device.queue.writeBuffer(this.vertexCounts, 0, data);
	}

	setDensities(data: Uint32Array) {
		device.queue.writeBuffer(this.density, 0, data);
	}

	setCommands(data: Uint32Array) {
		device.queue.writeBuffer(this.commands, 0, data);
	}

	destroy() {
		// Destroy all GPU buffers to free memory
		this.voxelData.destroy();
		this.vertexCounts.destroy();
		this.commands.destroy();
		this.vertices.destroy();
		this.normals.destroy();
		this.materialColors.destroy();
		this.colors.destroy();
		this.density.destroy();
		this.light.destroy();
	}
}