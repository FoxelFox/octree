import {Chunk} from './chunk';
import {camera, device, gpu, gridSize} from '../index';
import {Cull} from '../pipeline/rendering/cull';
import {Mesh} from '../pipeline/generation/mesh';
import {Block} from '../pipeline/rendering/block';
import {Light} from '../pipeline/rendering/light';
import {Noise} from '../pipeline/generation/noise';
import {VoxelEditorHandler} from '../ui/voxel-editor';
import {VoxelEditor} from '../pipeline/generation/voxel_editor';

export class Streaming {
	grid = new Map<number, Chunk>();
	generationQueue = new Array<Chunk>();
	generatedChunks = new Array<Chunk>();

	noise = new Noise();
	light = new Light();
	block = new Block();
	mesh = new Mesh();
	cull = new Cull();

	voxelEditor = new VoxelEditor(this.block, this.mesh, this.light);
	voxelEditorHandler = new VoxelEditorHandler(gpu, this.voxelEditor);

	chunk: Chunk;

	get cameraPositionInGridSpace(): number[] {
		return [
			Math.floor(camera.position[0] / gridSize),
			Math.floor(camera.position[1] / gridSize),
			Math.floor(camera.position[2] / gridSize),
		];
	}

	map3D1D(p: number[]): number {
		return p[0] + 16384 * (p[1] + 16384 * p[2]);
	}

	init() {
		console.log('Running one-time octree generation and compaction...');
		const setupEncoder = device.createCommandEncoder();
		this.chunk = new Chunk(0, [0, 0, 0]);

		this.noise.registerChunk(this.chunk);
		this.mesh.registerChunk(this.chunk);
		this.cull.registerChunk(this.chunk);
		this.light.registerChunk(this.chunk);
		this.block.registerChunk(this.chunk);
		this.voxelEditor.registerChunk(this.chunk);


		this.noise.update(setupEncoder, this.chunk);

		device.queue.submit([setupEncoder.finish()]);


		// Generate distance field from voxel data
		console.log('Generating distance field...');
		const encoder = device.createCommandEncoder();

		this.mesh.update(encoder, this.chunk);
		this.light.update(encoder, this.chunk);
		device.queue.submit([encoder.finish()]);

		this.mesh.afterUpdate();
		this.light.afterUpdate();

		console.log('Setup complete.');
	}

	update(updateEncoder: GPUCommandEncoder) {
		const center = this.cameraPositionInGridSpace;
		const index = this.map3D1D(center);

		// Update lighting and culling every frame (async on GPU)
		this.light.update(updateEncoder, this.chunk);
		this.cull.update(updateEncoder, this.chunk);
		this.block.update(updateEncoder, this.chunk);

		device.queue.submit([updateEncoder.finish()]);

		this.block.afterUpdate();
		this.mesh.afterUpdate();
		this.cull.afterUpdate();
		this.light.afterUpdate();
	}
}
