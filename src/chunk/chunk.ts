export class Chunk {
	id: number[]
	meshes: GPUBuffer;
	commands: GPUBuffer;
	density: GPUBuffer;
	voxelData: GPUBuffer; // noise and color
	count: number = 0;
	indices: Uint32Array;
	light: GPUBuffer;
}