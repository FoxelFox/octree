export class Chunk {
	id: number
	position: number[]
	meshes: GPUBuffer;
	commands: GPUBuffer;
	density: GPUBuffer;
	voxelData: GPUBuffer; // noise and color
	count: number = 0;
	indices: Uint32Array;
	light: GPUBuffer;

	constructor(id: number, position: number[]) {
		this.id = id;
		this.position = position;
	}

}