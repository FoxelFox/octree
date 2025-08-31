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

fn test_density_occlusion(block_pos: vec3<u32>) -> bool {
	// Extract camera position from inverse view matrix
	let camera_pos = context.inverse_view[3].xyz;
	
	// Get AABB for this block
	let aabb = get_block_aabb(block_pos);
	
	// Define the 8 corners of the block
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

	// Step size of one block (8x8x8 units)
	let step_size = 8.0;
	let density_threshold = 2048u; // Half-filled blocks start occluding
	let max_steps = 32u; // Reasonable limit for performance
	
	var min_accumulated_density = 4294967295u; // Max u32 value

	// Test occlusion for each corner
	for (var corner_idx = 0u; corner_idx < 8u; corner_idx++) {
		let corner = corners[corner_idx];
		
		// Ray direction from corner to camera
		let ray_dir = normalize(camera_pos - corner);
		
		// Start from current corner position
		var current_pos = corner;
		var accumulated_density = 0u;
		
		// Traverse toward camera
		for (var step = 0u; step < max_steps; step++) {
			// Move one block toward camera
			current_pos += ray_dir * step_size;
			
			// Convert world position back to block coordinates
			let test_block_pos_f = floor(current_pos / 8.0);
			
			// Check bounds before converting to unsigned
			let grid_size = context.grid_size / COMPRESSION;
			if (any(test_block_pos_f < vec3<f32>(0.0)) || any(test_block_pos_f >= vec3<f32>(f32(grid_size)))) {
				break;
			}
			
			let test_block_pos = vec3<u32>(test_block_pos_f);
			
			// Get density value for this block
			let block_index = to1DSmall(test_block_pos);
			let block_density = density[block_index];
			
			accumulated_density += block_density;
			
			// If we're close to camera, stop checking
			let distance_to_camera = length(camera_pos - current_pos);
			if (distance_to_camera < step_size) {
				break;
			}
		}
		
		// Track minimum density across all corners
		min_accumulated_density = min(min_accumulated_density, accumulated_density);
	}
	
	// Only cull if ALL corners are sufficiently occluded
	return min_accumulated_density >= density_threshold;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {

	let index = to1DSmall(id);
	let view_proj = context.perspective * context.view;


	let mesh = meshes[index];

	if (mesh.vertexCount > 0) {

		// Frustum culling: use fixed 8x8x8 block size
		let block_pos = vec3<u32>(id);
		let aabb = get_block_aabb(block_pos);
		if (test_aabb_frustum(aabb, view_proj)) {
			// Density occlusion culling: check if path to camera is blocked
			if (!test_density_occlusion(block_pos)) {
				let pointer = atomicAdd(&counter, 1u);
				indices[pointer] = index;
			}
		}
	}

}