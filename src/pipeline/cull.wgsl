#import "../data/context.wgsl"

struct Mesh {
	vertexCount: u32,
	vertices: array<vec4<f32>, 2048>, // worst case is way larger than 2048
}

// Input
@group(0) @binding(1) var<storage, read> meshes: array<Mesh>;
@group(0) @binding(3) var<storage, read> density: array<u32>;
@group(1) @binding(0) var<uniform> context: Context;

// Output
@group(0) @binding(0) var<storage, read_write> counter: atomic<u32>;
@group(0) @binding(2) var<storage, read_write> indices: array<u32>;

struct AABB {
	min: vec3<f32>,
	max: vec3<f32>,
}

fn get_block_aabb(block_pos: vec3<u32>) -> AABB {
	let world_pos = vec3<f32>(block_pos) * 8.0;
	var aabb: AABB;
	aabb.min = world_pos;
	aabb.max = world_pos + vec3<f32>(8.0);
	return aabb;
}

fn test_aabb_frustum(aabb: AABB, view_proj: mat4x4<f32>) -> bool {
	let corners = array<vec3<f32>, 8>(
		vec3<f32>(aabb.min.x, aabb.min.y, aabb.min.z),
		vec3<f32>(aabb.max.x, aabb.min.y, aabb.min.z),
		vec3<f32>(aabb.min.x, aabb.max.y, aabb.min.z),
		vec3<f32>(aabb.max.x, aabb.max.y, aabb.min.z),
		vec3<f32>(aabb.min.x, aabb.min.y, aabb.max.z),
		vec3<f32>(aabb.max.x, aabb.min.y, aabb.max.z),
		vec3<f32>(aabb.min.x, aabb.max.y, aabb.max.z),
		vec3<f32>(aabb.max.x, aabb.max.y, aabb.max.z)
	);
	
	for (var i = 0u; i < 8u; i++) {
		let clip_pos = view_proj * vec4<f32>(corners[i], 1.0);
		let ndc = clip_pos.xyz / clip_pos.w;
		
		// If any corner is inside frustum, AABB intersects
		if (all(ndc >= vec3<f32>(-1.0)) && all(ndc <= vec3<f32>(1.0)) && clip_pos.w > 0.0) {
			return true;
		}
	}
	
	return false;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {

	// fuck you compiler
	let co = context;
	let index = to1DSmall(id);
	let view_proj = co.perspective * co.view;
	let x = density[0];


	let mesh = meshes[index];

	if (mesh.vertexCount > 0) {

		// Frustum culling: use fixed 8x8x8 block size
		let block_pos = vec3<u32>(id);
		let aabb = get_block_aabb(block_pos);
		if (test_aabb_frustum(aabb, view_proj)) {
			let pointer = atomicAdd(&counter, 1u);
			indices[pointer] = index;
		}
	}

}