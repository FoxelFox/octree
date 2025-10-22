import {Request, Result} from "./generation/message";

import init, {generate_mesh, get_memory} from "src/my-lib/pkg"

let initialized = false;
let memory

onmessage = async (e: MessageEvent<Request>) => {

	if (!initialized) {
		await init();
		memory = get_memory();
		initialized = true;
	}

	let result: Result = {
		id: e.data.id
	}


	console.log("TEST")

	switch (e.data.operation) {
		case 'noise_for_chunk':
			const chunk = generate_mesh(e.data.args[0], e.data.args[1], e.data.args[2], e.data.args[3]);

			result.data = {};

			// Create views into WASM memory (zero-copy)
			result.data.vertices = new Float32Array(memory.buffer, chunk.vertices(), chunk.vertices_len());
			result.data.normals = new Float32Array(memory.buffer, chunk.normals(), chunk.normals_len());
			result.data.colors = new Uint32Array(memory.buffer, chunk.colors(), chunk.colors_len());
			result.data.material_colors = new Uint32Array(memory.buffer, chunk.material_colors(), chunk.material_colors_len());
			result.data.commands = new Uint32Array(memory.buffer, chunk.commands(), chunk.commands_len() * 4);
			result.data.densities = new Uint32Array(memory.buffer, chunk.densities(), chunk.density_len());
			result.data.vertex_counts = new Uint32Array(memory.buffer, chunk.vertex_counts(), chunk.vertex_counts_len());

			break;
	}
	postMessage(result);
};