import {Request, Result} from "./generation/message";

import init, {generate_mesh_direct} from "src/my-lib/pkg"

let initialized = false;

onmessage = async (e: MessageEvent<Request>) => {

	if (!initialized) {
		await init();
		initialized = true;
	}

	switch (e.data.operation) {
		case 'noise_for_chunk':
			// Generate mesh and get typed arrays directly from Rust
			const meshResult = generate_mesh_direct(
				e.data.args[0],
				e.data.args[1],
				e.data.args[2],
				e.data.args[3]
			);

			// Copy from WASM memory to JS-owned typed arrays
			const result: Result = {
				id: e.data.id,
				vertices: new Float32Array(meshResult.vertices),
				normals: new Float32Array(meshResult.normals),
				colors: new Uint32Array(meshResult.colors),
				material_colors: new Uint32Array(meshResult.material_colors),
				commands: new Uint32Array(meshResult.commands),
				densities: new Uint32Array(meshResult.densities),
				vertex_counts: new Uint32Array(meshResult.vertex_counts),
			};

			// Send back data (will be transferred efficiently via structured clone)
			postMessage(result);
			break;
	}
};