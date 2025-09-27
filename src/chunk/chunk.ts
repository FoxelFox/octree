export class Chunk {
	id: number
	position: number[]
	voxelData: GPUBuffer;    // noise and color
	vertexCounts: GPUBuffer; // just for the cull pipeline to fast ignore empty meshlets
	commands: GPUBuffer;     // indirect draw commands

	vertices: GPUBuffer;
	normals: GPUBuffer;
	colors: GPUBuffer;
	indices: Uint32Array;    // indices for meshlets that are actually needed to be rendered
	density: GPUBuffer;      // used for density occlusion culling and light blocker


	light: GPUBuffer;		// double buffered light data
	nextLight: GPUBuffer;   // double buffered light data


	constructor(id: number, position: number[]) {
		this.id = id;
		this.position = position;
	}

}