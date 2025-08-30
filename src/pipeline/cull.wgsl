#import "../data/context.wgsl"

struct Mesh {
	vertexCount: u32,
	normal: vec3<f32>,
	vertices: array<vec4<f32>, 384>, // worst case is way larger than 2048
}

// Input
@group(0) @binding(1) var<storage, read> meshes: array<Mesh>;
@group(1) @binding(0) var<uniform> context: Context;

// Output
@group(0) @binding(0) var<storage, read_write> counter: atomic<u32>;
@group(0) @binding(2) var<storage, read_write> indices: array<u32>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {

	// fuck you compiler
	let co = context;
	let index = to1DSmall(id);

	for (var n = 0u; n < 6u; n++) {
		let i = index * 6 + n;
		let mesh = meshes[i];

		if (mesh.vertexCount > 0) {
			// Extract camera direction from inverse view matrix (forward vector)
			let camera_forward = normalize(co.inverse_view[2].xyz);
			
			// Cull back-facing meshes (dot product > 0 means facing camera)
			if (dot(mesh.normal, camera_forward) > 0.0) {
				let pointer = atomicAdd(&counter, 1u);
				indices[pointer] = i;
			}
		}
	}





}
