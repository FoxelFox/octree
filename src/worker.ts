import {Request, Result} from "./generation/message";

import init, {generate_mesh, get_memory, noise_for_chunk} from "src/my-lib/pkg"

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
			result.data = noise_for_chunk(e.data.args[0], e.data.args[1], e.data.args[2], e.data.args[3]);

			const chunk = generate_mesh(e.data.args[0], e.data.args[1], e.data.args[2], e.data.args[3]);

			// Create views into WASM memory (zero-copy)

			const vertices = new Float32Array(
				memory.buffer,
				chunk.vertices(),
				chunk.vertices_len()
			);

			
			break;
	}
	postMessage(result);
};