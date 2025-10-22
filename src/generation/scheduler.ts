import {Result} from "./message";

interface Task {
	id: number
	operation: string
	args: any[]
	resolve: (result: any) => void
	sharedBuffers: {
		vertices: SharedArrayBuffer
		normals: SharedArrayBuffer
		colors: SharedArrayBuffer
		material_colors: SharedArrayBuffer
		commands: SharedArrayBuffer
		densities: SharedArrayBuffer
		vertex_counts: SharedArrayBuffer
	}
}

interface BufferSet {
	vertices: SharedArrayBuffer
	normals: SharedArrayBuffer
	colors: SharedArrayBuffer
	material_colors: SharedArrayBuffer
	commands: SharedArrayBuffer
	densities: SharedArrayBuffer
	vertex_counts: SharedArrayBuffer
}

export class Scheduler {

	idCounter = 0;
	idle: Worker[] = [];

	activeTasks: Map<number, Task> = new Map();
	queue: Task[] = [];

	// Estimate max buffer sizes (these can be adjusted based on actual needs)
	// For a 256^3 chunk with compression 8: (256/8)^3 = 32768 meshlets
	// Each meshlet can have ~512 vertices max
	private readonly MAX_VERTICES = 32768 * 512;
	private readonly MAX_MESHLETS = 32768;

	// Buffer pool - reuse SharedArrayBuffers instead of allocating new ones
	private bufferPool: BufferSet[] = [];

	constructor() {
		// Pre-allocate buffer sets (one per worker to avoid contention)
		for (let i = 0; i < navigator.hardwareConcurrency; i++) {
			this.bufferPool.push(this.createBufferSet());
		}

		for (let i = 0; i < navigator.hardwareConcurrency; i++) {

			const worker = new Worker("./worker.js", {type: 'module'});

			worker.onmessage = (res) => {
				const r = res.data as Result;
				const task = this.activeTasks.get(r.id);

				this.activeTasks.delete(r.id);

				// Construct views from shared buffers and copy data
				const buffers = task.sharedBuffers;
				const data = {
					vertices: new Float32Array(buffers.vertices, 0, r.metadata.verticesLength).slice(),
					normals: new Float32Array(buffers.normals, 0, r.metadata.normalsLength).slice(),
					colors: new Uint32Array(buffers.colors, 0, r.metadata.colorsLength).slice(),
					material_colors: new Uint32Array(buffers.material_colors, 0, r.metadata.materialColorsLength).slice(),
					commands: new Uint32Array(buffers.commands, 0, r.metadata.commandsLength).slice(),
					densities: new Uint32Array(buffers.densities, 0, r.metadata.densitiesLength).slice(),
					vertex_counts: new Uint32Array(buffers.vertex_counts, 0, r.metadata.vertexCountsLength).slice(),
				};
				task.resolve(data);

				// Return buffers to pool
				this.bufferPool.push(task.sharedBuffers);

				this.idle.push(worker);
				this.update();
			}

			this.idle.push(worker);

		}
	}

	private createBufferSet(): BufferSet {
		return {
			vertices: new SharedArrayBuffer(this.MAX_VERTICES * 4 * 4), // vec4<f32>
			normals: new SharedArrayBuffer(this.MAX_VERTICES * 4 * 4),  // vec4<f32> (with padding)
			colors: new SharedArrayBuffer(this.MAX_VERTICES * 4),       // u32
			material_colors: new SharedArrayBuffer(this.MAX_VERTICES * 4), // u32
			commands: new SharedArrayBuffer(this.MAX_MESHLETS * 4 * 4),   // 4 u32s per command
			densities: new SharedArrayBuffer(this.MAX_MESHLETS * 4),      // u32
			vertex_counts: new SharedArrayBuffer(this.MAX_MESHLETS * 4),  // u32
		};
	}


	async work(operation: string, args: any): Promise<any> {
		return new Promise(resolve => {
			const id = this.idCounter++;

			// Get SharedArrayBuffers from pool (or create new if pool is empty)
			const sharedBuffers = this.bufferPool.pop() || this.createBufferSet();

			const task: Task = {
				id,
				operation,
				args,
				resolve,
				sharedBuffers
			}

			this.queue.push(task);
			this.update();
		});
	}

	update() {
		while (this.queue.length && this.idle.length) {
			const worker = this.idle.shift();
			const task = this.queue.shift();
			this.activeTasks.set(task.id, task);
			worker.postMessage({
				id: task.id,
				operation: task.operation,
				args: task.args,
				sharedBuffers: task.sharedBuffers
			});
		}
	}


}