import {compression, device, gridSize} from "../index";

export class Chunk {
	id: number
	position: number[]
	voxelData: GPUBuffer;    // density (f32) + color (u32) = 8 bytes per voxel
	vertexCounts: GPUBuffer; // just for the cull pipeline to fast ignore empty meshlets
	commands: GPUBuffer;     // indirect draw commands

	vertices: GPUBuffer;
	normals: GPUBuffer;
	colors: GPUBuffer;
	density: GPUBuffer;      // used for density occlusion culling and light blocker
	indices: Uint32Array;    // indices for meshlets that are actually needed to be rendered


	light: GPUBuffer;		// double buffered light data
	nextLight: GPUBuffer;   // double buffered light data


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


		this.voxelData = device.createBuffer({
			label: 'Voxel Data',
			size,
			usage:
				GPUBufferUsage.STORAGE |
				GPUBufferUsage.COPY_DST |
				GPUBufferUsage.COPY_SRC,
		});

		// Separate buffers for mesh data
		this.vertexCounts = device.createBuffer({
			size: 4 * sSize3, // u32 = 4 bytes each
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
		});

		this.vertices = device.createBuffer({
			size: 8 * maxVertices, // vec4<f16> = 8 bytes each
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
		});

		this.normals = device.createBuffer({
			size: 8 * maxVertices, // vec3<f16> = 8 bytes each (with padding)
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
		});

		this.colors = device.createBuffer({
			size: 4 * maxVertices, // u32 = 4 bytes each
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
		});

		this.commands = device.createBuffer({
			size: 16 * sSize3,
			usage:
				GPUBufferUsage.STORAGE |
				GPUBufferUsage.COPY_SRC |
				GPUBufferUsage.INDIRECT,
		});

		this.density = device.createBuffer({
			size: 4 * sSize3,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
		});


		// Create both light buffers: stores light intensity (R) and shadow factor (G) for each compressed cell
		// Format: RG32Float - R = light intensity, G = shadow factor (0.0 = fully lit, 1.0 = fully shadowed)
		const bufferConfig = {
			size: sSize3 * 8, // 2 * 4 bytes per cell (RG32Float)
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
		};
		this.light = device.createBuffer(bufferConfig);
		this.nextLight = device.createBuffer(bufferConfig);
	}

}