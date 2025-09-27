import { Chunk } from './chunk';
import { camera, device, gpu, gridSize } from '../index';
import { Cull } from '../pipeline/rendering/cull';
import { Mesh } from '../pipeline/generation/mesh';
import { Block } from '../pipeline/rendering/block';
import { Light } from '../pipeline/rendering/light';
import { Noise } from '../pipeline/generation/noise';
import { VoxelEditorHandler } from '../ui/voxel-editor';
import { VoxelEditor } from '../pipeline/generation/voxel_editor';

export class Streaming {
	grid = new Map<number, Chunk>();
	generationQueue = new Array<Chunk>();
	generatedChunks = new Array<Chunk>();

	noise = new Noise();
	light = new Light();
	block = new Block();
	mesh = new Mesh();
	cull = new Cull();
	voxelEditorHandler: VoxelEditorHandler;
	voxelEditor: VoxelEditor;

	constructor() {
		this.block.mesh = this.mesh;
		this.block.cull = this.cull;
		this.block.light = this.light;
	}

	map3D1D(p: number[]): number {
		return p[0] + 16384 * (p[1] + 16384 * p[2]);
	}

	get cameraPositionInGridSpace(): number[] {
		return [
			Math.floor(camera.position[0] / gridSize),
			Math.floor(camera.position[1] / gridSize),
			Math.floor(camera.position[2] / gridSize),
		];
	}

	init() {
		console.log('Running one-time octree generation and compaction...');
		const setupEncoder = device.createCommandEncoder();
		this.noise.update(setupEncoder);

		device.queue.submit([setupEncoder.finish()]);

		this.mesh.init(this.noise);
		this.cull.init(this.noise, this.mesh);
		this.light.init(this.mesh);

		// Generate distance field from voxel data
		console.log('Generating distance field...');
		const encoder = device.createCommandEncoder();

		this.mesh.update(encoder);
		this.light.update(encoder, this.mesh);
		device.queue.submit([encoder.finish()]);

		this.mesh.afterUpdate();
		this.light.afterUpdate();

		// Initialize voxel editor after all systems are ready
		this.voxelEditor = new VoxelEditor(
			this.block,
			this.noise,
			this.mesh,
			this.cull,
			this.light
		);
		this.voxelEditorHandler = new VoxelEditorHandler(gpu, this.voxelEditor);

		console.log('Setup complete.');
	}
	
	update(updateEncoder: GPUCommandEncoder) {
		const center = this.cameraPositionInGridSpace;
		const index = this.map3D1D(center);

		if (!this.grid.has(index)) {
			const chunk = new Chunk(index, center);
			this.generationQueue.push(chunk);
			this.grid.set(index, chunk);
		}

		// Update lighting and culling every frame (async on GPU)
		this.light.update(updateEncoder, this.mesh);
		this.cull.update(updateEncoder);
		this.block.update(updateEncoder);

		device.queue.submit([updateEncoder.finish()]);

		this.block.afterUpdate();
		this.mesh.afterUpdate();
		this.cull.afterUpdate();
		this.light.afterUpdate();
	}
}
