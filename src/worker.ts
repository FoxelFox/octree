import {Request, Result} from "./generation/message";

import init, {generate_mesh_to_shared} from "src/my-lib/pkg"

let initialized = false;

onmessage = async (e: MessageEvent<Request>) => {

	if (!initialized) {
		await init();
		initialized = true;
	}

	let result: Result = {
		id: e.data.id
	}

	switch (e.data.operation) {
		case 'noise_for_chunk':
			// Create views into SharedArrayBuffers
			const sharedVertices = new Float32Array(e.data.sharedBuffers.vertices);
			const sharedNormals = new Float32Array(e.data.sharedBuffers.normals);
			const sharedColors = new Uint32Array(e.data.sharedBuffers.colors);
			const sharedMaterialColors = new Uint32Array(e.data.sharedBuffers.material_colors);
			const sharedCommands = new Uint32Array(e.data.sharedBuffers.commands);
			const sharedDensities = new Uint32Array(e.data.sharedBuffers.densities);
			const sharedVertexCounts = new Uint32Array(e.data.sharedBuffers.vertex_counts);

			// Generate mesh directly into SharedArrayBuffers (zero-copy!)
			const metadata = generate_mesh_to_shared(
				e.data.args[0],
				e.data.args[1],
				e.data.args[2],
				e.data.args[3],
				sharedVertices,
				sharedNormals,
				sharedColors,
				sharedMaterialColors,
				sharedCommands,
				sharedDensities,
				sharedVertexCounts
			);

			// Send back metadata only (no data copy via postMessage)
			result.metadata = {
				verticesLength: metadata.vertices_length,
				normalsLength: metadata.normals_length,
				colorsLength: metadata.colors_length,
				materialColorsLength: metadata.material_colors_length,
				commandsLength: metadata.commands_length,
				densitiesLength: metadata.densities_length,
				vertexCountsLength: metadata.vertex_counts_length,
			};

			break;
	}
	postMessage(result);
};