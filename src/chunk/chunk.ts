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


	constructor(id: number, position: number[]) {
		this.id = id;
		this.position = position;

		// Allocate 257³ voxels (gridSize + 1) to handle chunk borders
		// Each voxel: density (f32) + color (u32) = 8 bytes
		const voxelGridSize = gridSize + 1;
		const size = Math.pow(voxelGridSize, 3) * 8;
		const sSize = gridSize / compression;
		const sSize3 = sSize * sSize * sSize;
		const maxVertices = sSize3 * 1536;

		const chunkLabel = `Chunk[${id}](${position[0]},${position[1]},${position[2]})`;

		this.voxelData = device.createBuffer({
			label: `${chunkLabel} Voxel Data`,
			size,
			usage:
				GPUBufferUsage.STORAGE |
				GPUBufferUsage.COPY_DST |
				GPUBufferUsage.COPY_SRC,
		});

		// Separate buffers for mesh data
		this.vertexCounts = device.createBuffer({
			label: `${chunkLabel} Vertex Counts`,
			size: 4 * sSize3, // u32 = 4 bytes each
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
		});

		// Start with small initial buffers - they will grow as needed
		const initialVertexCount = 1024;

		this.vertices = device.createBuffer({
			label: `${chunkLabel} Vertices`,
			size: 8 * initialVertexCount, // vec4<f16> = 8 bytes each
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
		});

		this.normals = device.createBuffer({
			label: `${chunkLabel} Normals`,
			size: 8 * initialVertexCount, // vec3<f16> = 8 bytes each (with padding)
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
		});

		this.materialColors = device.createBuffer({
			label: `${chunkLabel} Material Colors`,
			size: 4 * initialVertexCount, // u32 = 4 bytes each
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
		});

		this.colors = device.createBuffer({
			label: `${chunkLabel} Lit Colors`,
			size: 4 * initialVertexCount, // u32 = 4 bytes each
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
		});

		this.commands = device.createBuffer({
			label: `${chunkLabel} Commands`,
			size: 16 * sSize3,
			usage:
				GPUBufferUsage.STORAGE |
				GPUBufferUsage.COPY_SRC |
				GPUBufferUsage.INDIRECT,
		});

		this.density = device.createBuffer({
			label: `${chunkLabel} Density`,
			size: 4 * sSize3,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
		});


		// Create light buffer: stores light intensity (R) and shadow factor (G) for each compressed cell
		// Format: RG32Float - R = light intensity, G = shadow factor (0.0 = fully lit, 1.0 = fully shadowed)
		this.light = device.createBuffer({
			label: `${chunkLabel} Light`,
			size: sSize3 * 8, // 2 * 4 bytes per cell (RG32Float)
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
		});
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