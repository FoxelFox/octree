#import "../../data/context.wgsl"


// Input
@group(0) @binding(1) var<storage, read> vertexCounts: array<u32>;
@group(0) @binding(3) var<storage, read> combined_density: array<u32>;
@group(1) @binding(0) var<uniform> context: Context;
@group(1) @binding(1) var<uniform> chunk_world_pos: vec4<i32>; // vec4 for Safari compatibility
@group(1) @binding(2) var<uniform> chunk_resolution: vec4<u32>; // vec4 for alignment, only .x is used

// Output
@group(0) @binding(0) var<storage, read_write> counter: atomic<u32>;
@group(0) @binding(2) var<storage, read_write> indices: array<u32>;

struct AABB {
	min: vec3<f32>,
	max: vec3<f32>,
}

const NEIGHBOR_RADIUS: i32 = 1;
const NEIGHBOR_DIAMETER: i32 = NEIGHBOR_RADIUS * 2 + 1;
const NEIGHBOR_LAYER: i32 = NEIGHBOR_DIAMETER * NEIGHBOR_DIAMETER;

struct DensitySample {
	value: u32,
	valid: bool,
}

fn chunk_block_count() -> u32 {
	return chunk_resolution.x / u32(COMPRESSION);
}

fn cells_per_chunk() -> u32 {
	let block_count = chunk_block_count();
	return block_count * block_count * block_count;
}

fn index_from_offset(offset: vec3<i32>) -> u32 {
	let translated = offset + vec3<i32>(NEIGHBOR_RADIUS);
	let index = translated.z * NEIGHBOR_LAYER + translated.y * NEIGHBOR_DIAMETER + translated.x;
	return u32(index);
}

fn sample_combined_density(world_pos: vec3<f32>) -> DensitySample {
	// Calculate the world-space size of each meshlet for this chunk
	let meshlet_count = chunk_block_count();
	let meshlet_world_size = 256.0 / f32(meshlet_count);

	let world_block_pos = vec3<i32>(floor(world_pos / meshlet_world_size));
	let chunk_block_origin = vec3<i32>(floor(vec3<f32>(chunk_world_pos.xyz) / meshlet_world_size));
	let relative = world_block_pos - chunk_block_origin;
	let block_count_i = i32(meshlet_count);
	let offset = vec3<i32>(floor(vec3<f32>(relative) / f32(block_count_i)));

	if (any(offset < vec3<i32>(-NEIGHBOR_RADIUS)) || any(offset > vec3<i32>(NEIGHBOR_RADIUS))) {
		return DensitySample(0u, false);
	}

	let local = relative - offset * block_count_i;
	if (any(local < vec3<i32>(0)) || any(local >= vec3<i32>(block_count_i))) {
		return DensitySample(0u, false);
	}

	let local_index = to1DSmall(vec3<u32>(local));
	let chunk_index = index_from_offset(offset);
	let base = chunk_index * cells_per_chunk();
	return DensitySample(combined_density[base + local_index], true);
}

fn get_block_aabb(block_pos: vec3<u32>) -> AABB {
	// Calculate world-space size of each meshlet
	// Chunk occupies 256 units, divided by number of meshlets per axis
	let meshlet_count = chunk_block_count();
	let meshlet_world_size = 256.0 / f32(meshlet_count);

	// Convert block position to world space
	let local_pos = vec3<f32>(block_pos) * meshlet_world_size;
	let world_pos = local_pos + vec3<f32>(chunk_world_pos.xyz);

	var aabb: AABB;
	aabb.min = world_pos;
	aabb.max = world_pos + vec3<f32>(meshlet_world_size);
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
	let camera_pos = context.inverse_view[3].xyz;
	let aabb = get_block_aabb(block_pos);
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

	// Step size should match the world-space size of each meshlet
	let meshlet_count = chunk_block_count();
	let step_size = 256.0 / f32(meshlet_count);
	let density_threshold = 1024u;
	let max_steps = meshlet_count;

	for (var corner_idx = 0u; corner_idx < 8u; corner_idx++) {
		let corner = corners[corner_idx];
		let ray_dir = normalize(camera_pos - corner);
		var current_pos = corner;
		var accumulated_density = 0u;

		for (var step = 0u; step < max_steps; step++) {
			current_pos += ray_dir * step_size;
			let sample = sample_combined_density(current_pos);
			if (!sample.valid) {
				break;
			}

			accumulated_density += sample.value;
			if (accumulated_density >= density_threshold) {
				break;
			}

			let distance_to_camera = length(camera_pos - current_pos);
			if (distance_to_camera < step_size) {
				break;
			}
		}

		if (accumulated_density < density_threshold) {
			return false;
		}
	}

	return true;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
	// Early exit if this thread is beyond the chunk's meshlet grid
	let meshlet_count = chunk_block_count();
	if (id.x >= meshlet_count || id.y >= meshlet_count || id.z >= meshlet_count) {
		return;
	}

	// Calculate 1D index using the chunk's actual meshlet count
	let index = id.z * meshlet_count * meshlet_count + id.y * meshlet_count + id.x;
	let view_proj = context.perspective * context.view;

	let vertexCount = vertexCounts[index];

	if (vertexCount > 0) {

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
