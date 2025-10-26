import {Result} from "./message";

interface Task {
	id: number
	operation: string
	args: any[]
	resolve: (result: any) => void
}

export class Scheduler {

	idCounter = 0;
	idle: Worker[] = [];

	activeTasks: Map<number, Task> = new Map();
	queue: Task[] = [];

	constructor() {

		for (let i = 0; i < navigator.hardwareConcurrency; i++) {

			const worker = new Worker("./worker.js", {type: 'module'});

			worker.onmessage = (res) => {
				const r = res.data as Result;
				const task = this.activeTasks.get(r.id);

				this.activeTasks.delete(r.id);

				// Data already comes as typed arrays from worker
				const data = {
					vertices: r.vertices,
					normals: r.normals,
					colors: r.colors,
					material_colors: r.material_colors,
					commands: r.commands,
					densities: r.densities,
					vertex_counts: r.vertex_counts,
					indices: r.indices,
				};
				task.resolve(data);

				this.idle.push(worker);
				this.update();
			}

			this.idle.push(worker);

		}
	}

	async work(operation: string, args: any): Promise<any> {
		return new Promise(resolve => {
			const id = this.idCounter++;

			const task: Task = {
				id,
				operation,
				args,
				resolve,
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
			});
		}
	}


}