export class Chunk {
	id: number[]
	meshes: GPUBuffer;
	commands: GPUBuffer;
	density: GPUBuffer;
	count: number = 0;
	indices: Uint32Array;
}