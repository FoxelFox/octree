#import "../data/context.wgsl"

struct Mesh {
	vertexCount: u32,
	vertices: array<vec4<f32>, 1536>, // worst case is way larger than 2048
}

// Input
@group(0) @binding(1) var<storage, read> mesh: array<Mesh>;
@group(1) @binding(0) var<uniform> context: Context;

// Output
@group(0) @binding(0) var<storage, read_write> counter: atomic<u32>;
@group(0) @binding(2) var<storage, read_write> indices: array<u32>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {

	// fuck you compiler
	let c = &counter;
	let m = mesh[0];
	let co = context;
	let i = indices[0];

	// cull by viewport
	let index = to1DSmall(id);




}
