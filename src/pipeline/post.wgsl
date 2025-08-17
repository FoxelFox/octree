#import "../data/context.wgsl"

struct CompactNode {
    firstChildOrData: u32,
    childMask: u32,
}

struct DistanceNode {
    distance: f32,
    material_id: u32,
}

const LEAF_BIT: u32 = 0x80000000u;
const INVALID_INDEX: u32 = 0xFFFFFFFFu;
const MAX_STACK: u32 = 21u; // must be >= context.max_depth + a small margin
const MAX_INTERSECTION_TESTS: u32 = 384; // maximum ray-AABB tests before early termination

// Distance field constants
const SDF_EPSILON: f32 = 0.001; // Surface detection threshold
const SDF_MAX_STEPS: u32 = 256u; // Maximum raymarching steps (increased)
const SDF_MAX_DISTANCE: f32 = 1000.0; // Maximum ray distance
const SDF_OVER_RELAXATION: f32 = 0.8; // Step size multiplier for better precision

@group(0) @binding(0) var <uniform> context: Context;
@group(0) @binding(1) var<storage, read> nodes: array<CompactNode>;
@group(0) @binding(2) var<storage, read> distance_field: array<DistanceNode>;
@group(1) @binding(0) var prevFrameTexture: texture_2d<f32>;
@group(1) @binding(1) var smpler: sampler;
@group(1) @binding(2) var prevWorldPosTexture: texture_2d<f32>;
@group(1) @binding(3) var blueNoiseTexture: texture_2d<f32>;

@vertex
fn main_vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  let c = context;
  const pos = array(
    vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
    vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
  );
  return vec4(pos[i], 0.0, 1.0);
}

// A more numerically robust slab ray-AABB intersection
fn ray_aabb_intersect(ray_o: vec3<f32>, ray_d: vec3<f32>, box_min: vec3<f32>, box_max: vec3<f32>) -> vec2<f32> {
    // Pre-calculate the inverse of the ray direction.
    // This is more efficient and central to the robust algorithm.
    let inv_dir = 1.0 / ray_d;

    // For each axis, calculate the `t` values for the two slab planes.
    // We use the sign of inv_dir to determine which plane is near and which is far
    // without needing a conditional swap, which is more stable.
    let t1 = (box_min - ray_o) * inv_dir;
    let t2 = (box_max - ray_o) * inv_dir;

    // Find the min and max for each axis pair
    let t_min_ax = min(t1, t2);
    let t_max_ax = max(t1, t2);

    // The final tmin is the largest of the per-axis tmin values.
    let tmin = max(t_min_ax.x, max(t_min_ax.y, t_min_ax.z));

    // The final tmax is the smallest of the per-axis tmax values.
    let tmax = min(t_max_ax.x, min(t_max_ax.y, t_max_ax.z));

    // A valid intersection requires tmin <= tmax.
    // If tmin > tmax, the ray misses the box. We return (1.0, 0.0) to signal a miss.
    if (tmin > tmax) {
        return vec2<f32>(1.0, 0.0);
    }

    return vec2<f32>(tmin, tmax);
}

// compute child AABB min given parent min, parent size and octant index
fn child_min_from_parent(parent_min: vec3<f32>, parent_size: f32, octant: u32) -> vec3<f32> {
    let half = parent_size * 0.5;
    let offset = vec3<f32>(
        f32((octant & 1u) != 0u),
        f32((octant & 2u) != 0u),
        f32((octant & 4u) != 0u)
    );
    return parent_min + offset * half;
}

struct RayCast {
	pos: vec3<f32>,
	normal: vec3<f32>,
	data: u32,
	steps: u32
}

// Blue noise lookup with tiling and temporal variation
fn sampleBlueNoise(uv: vec2<f32>, offset: f32) -> f32 {
    // Tile the blue noise texture and add temporal variation
    let tiled_uv = fract(uv * 8.0 + vec2<f32>(offset * 0.1));
    let tex_coord = vec2<i32>(tiled_uv * 64.0); // 64 is the blue noise texture size
    let noise = textureLoad(blueNoiseTexture, tex_coord, 0).r;
    
    // Add temporal jitter using frame-based offset
    return fract(noise + offset);
}

// Cast a shadow ray to check occlusion between hit point and light source
fn cast_shadow_ray(from_pos: vec3<f32>, to_pos: vec3<f32>, grid_size: u32) -> bool {
    let shadow_dir = normalize(to_pos - from_pos);
    let shadow_dist = distance(to_pos, from_pos);
    
    // Offset start position slightly along normal to avoid self-intersection
    let shadow_start = from_pos + shadow_dir * 0.001;
    
    var shadow_hit: RayCast;
    
    // Choose raycast method based on render mode
    if (context.render_mode >= 2u) {
        // Use distance field raymarching
        shadow_hit = raycast_distance_field(shadow_start, shadow_dir);
    } else {
        // Use octree traversal
        shadow_hit = raycast_octree_stack(shadow_start, shadow_dir, grid_size);
    }
    
    // Check if we hit something before reaching the light
    if (shadow_hit.pos.x >= -0.001) {
        let hit_dist = distance(shadow_hit.pos, shadow_start);
        return hit_dist < (shadow_dist - 0.001);
    }
    
    return false;
}

// Generate a random direction within a cone for soft shadow sampling
fn random_cone_direction(seed: f32, cone_angle: f32, light_dir: vec3<f32>, pixel_uv: vec2<f32>) -> vec3<f32> {
    // Use blue noise for better distribution
    let frame_offset = context.random_seed + seed;
    let a = sampleBlueNoise(pixel_uv, frame_offset);
    let b = sampleBlueNoise(pixel_uv + vec2<f32>(0.5, 0.3), frame_offset + 1.7);
    
    // Convert to spherical coordinates within cone
    let theta = a * 2.0 * 3.14159;
    let phi = acos(1.0 - b * (1.0 - cos(cone_angle)));
    
    // Create orthonormal basis around light direction
    let w = light_dir;
    let cross_vec = select(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), abs(w.x) <= 0.1);
    let u = normalize(cross(cross_vec, w));
    let v = cross(w, u);
    
    // Convert spherical to cartesian in local basis
    let sin_phi = sin(phi);
    let local_dir = vec3(sin_phi * cos(theta), sin_phi * sin(theta), cos(phi));
    
    // Transform to world space
    return normalize(u * local_dir.x + v * local_dir.y + w * local_dir.z);
}

// Calculate soft shadows by sampling multiple rays within light area
fn calculate_soft_shadow(hit_pos: vec3<f32>, light_pos: vec3<f32>, light_size: f32, grid_size: u32, pixel_uv: vec2<f32>) -> f32 {
    var shadow_factor = 0.0;
    let seed = dot(hit_pos, vec3(12.9898, 78.233, 37.719)) + context.random_seed;

	// Generate random point on light area
	let light_dir = normalize(light_pos - hit_pos);
	let random_dir = random_cone_direction(seed, light_size, light_dir, pixel_uv);
	let sample_light_pos = light_pos + random_dir * light_size * 50.0; // Scale factor for light area

	if (!cast_shadow_ray(hit_pos, sample_light_pos, grid_size)) {
		shadow_factor += 1.0;
	}
    
    return shadow_factor;
}

// Generate random direction in hemisphere above surface normal
fn random_hemisphere_direction(seed: f32, normal: vec3<f32>, pixel_uv: vec2<f32>) -> vec3<f32> {
    // Use blue noise for better distribution
    let frame_offset = context.random_seed + seed;
    let a = sampleBlueNoise(pixel_uv + vec2<f32>(0.2, 0.7), frame_offset);
    let b = sampleBlueNoise(pixel_uv + vec2<f32>(0.8, 0.1), frame_offset + 2.3);
    
    // Generate uniform distribution on hemisphere
    let theta = a * 2.0 * 3.14159;
    let phi = acos(sqrt(b)); // Cosine-weighted distribution
    
    // Create orthonormal basis around normal
    let w = normal;
    let cross_vec = select(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), abs(w.y) > 0.9);
    let u = normalize(cross(cross_vec, w));
    let v = cross(w, u);
    
    // Convert spherical to cartesian in local basis
    let sin_phi = sin(phi);
    let local_dir = vec3(sin_phi * cos(theta), sin_phi * sin(theta), cos(phi));
    
    // Transform to world space
    return normalize(u * local_dir.x + v * local_dir.y + w * local_dir.z);
}

// Calculate sky light contribution by sampling hemisphere
fn calculate_sky_light(hit_pos: vec3<f32>, normal: vec3<f32>, grid_size: u32, samples: u32, pixel_uv: vec2<f32>) -> f32 {
    var sky_factor = 0.0;
    let base_seed = dot(hit_pos, vec3(17.531, 41.234, 63.719)) + context.random_seed;
    
    for (var i: u32 = 0u; i < samples; i++) {
        let seed = base_seed + f32(i) * 9.234;
        let sky_dir = random_hemisphere_direction(seed, normal, pixel_uv);
        
        var sky_hit: RayCast;
        
        // Choose raycast method based on render mode
        if (context.render_mode >= 2u) {
            // Use distance field raymarching
            sky_hit = raycast_distance_field(hit_pos + normal * 0.001, sky_dir);
        } else {
            // Use octree traversal
            sky_hit = raycast_octree_stack(hit_pos + normal * 0.001, sky_dir, grid_size);
        }
        
        if (sky_hit.pos.x < -0.001) {
            // Ray escaped to sky
            sky_factor += 1.0;
        }
    }
    
    return sky_factor / f32(samples);
}

// Trilinear interpolated distance field sampling for smooth results
fn sample_distance_field(pos: vec3<f32>) -> f32 {
    let gs_f = f32(context.grid_size);
    
    // Check if position is outside grid bounds
    if (any(pos < vec3<f32>(0.0)) || any(pos >= vec3<f32>(gs_f - 1.0))) {
        return 1.0; // Outside grid = empty space
    }
    
    // Get the eight surrounding grid points for trilinear interpolation
    let grid_pos = clamp(pos, vec3<f32>(0.0), vec3<f32>(gs_f - 1.01));
    let p0 = floor(grid_pos);
    let p1 = p0 + vec3<f32>(1.0);
    let t = grid_pos - p0;
    
    let gs = context.grid_size;
    
    // Sample the 8 corners of the cube
    let i000 = u32(p0.z) * gs * gs + u32(p0.y) * gs + u32(p0.x);
    let i001 = u32(p0.z) * gs * gs + u32(p0.y) * gs + u32(p1.x);
    let i010 = u32(p0.z) * gs * gs + u32(p1.y) * gs + u32(p0.x);
    let i011 = u32(p0.z) * gs * gs + u32(p1.y) * gs + u32(p1.x);
    let i100 = u32(p1.z) * gs * gs + u32(p0.y) * gs + u32(p0.x);
    let i101 = u32(p1.z) * gs * gs + u32(p0.y) * gs + u32(p1.x);
    let i110 = u32(p1.z) * gs * gs + u32(p1.y) * gs + u32(p0.x);
    let i111 = u32(p1.z) * gs * gs + u32(p1.y) * gs + u32(p1.x);
    
    let d000 = distance_field[i000].distance;
    let d001 = distance_field[i001].distance;
    let d010 = distance_field[i010].distance;
    let d011 = distance_field[i011].distance;
    let d100 = distance_field[i100].distance;
    let d101 = distance_field[i101].distance;
    let d110 = distance_field[i110].distance;
    let d111 = distance_field[i111].distance;
    
    // Trilinear interpolation
    let d00 = mix(d000, d001, t.x);
    let d01 = mix(d010, d011, t.x);
    let d10 = mix(d100, d101, t.x);
    let d11 = mix(d110, d111, t.x);
    
    let d0 = mix(d00, d01, t.y);
    let d1 = mix(d10, d11, t.y);
    
    return mix(d0, d1, t.z);
}

// Calculate normal using central differences (gradient of distance field)
fn calculate_sdf_normal(pos: vec3<f32>) -> vec3<f32> {
    let h = 0.01; // Much smaller offset for smooth gradients
    let dx = sample_distance_field(pos + vec3<f32>(h, 0.0, 0.0)) - sample_distance_field(pos - vec3<f32>(h, 0.0, 0.0));
    let dy = sample_distance_field(pos + vec3<f32>(0.0, h, 0.0)) - sample_distance_field(pos - vec3<f32>(0.0, h, 0.0));
    let dz = sample_distance_field(pos + vec3<f32>(0.0, 0.0, h)) - sample_distance_field(pos - vec3<f32>(0.0, 0.0, h));
    return normalize(vec3<f32>(dx, dy, dz));
}

// Raymarching implementation for distance fields
fn raycast_distance_field(ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> RayCast {
    var result: RayCast;
    result.pos = vec3<f32>(-1.0);
    result.data = 0u;
    result.steps = 0u;
    
    var t = 0.0;
    let epsilon = context.sdf_epsilon;
    let max_steps = context.sdf_max_steps;
    let over_relaxation = context.sdf_over_relaxation;
    
    var prev_dist = 1.0; // Initialize as outside
    
    for (var i: u32 = 0u; i < max_steps; i++) {
        let pos = ray_origin + ray_dir * t;
        let dist = sample_distance_field(pos);
        
        result.steps = i;
        
        // Check for transition from empty to solid (front surface)
        if (i > 0u && prev_dist >= 0.0 && dist < 0.0) {
            result.pos = pos;
            
            // Use distance field gradient for smooth normals
            result.normal = calculate_sdf_normal(pos);
            
            let grid_pos = clamp(pos, vec3<f32>(0.0), vec3<f32>(f32(context.grid_size) - 1.0));
            let grid_i = min(vec3<u32>(round(grid_pos)), vec3<u32>(context.grid_size - 1u));
            let gs = context.grid_size;
            let index = grid_i.z * gs * gs + grid_i.y * gs + grid_i.x;
            
            result.data = distance_field[index].material_id;
            
            return result;
        }
        
        // Adaptive stepping: use distance as step size with much smaller minimum step
        let step_size = max(abs(dist) * over_relaxation, 0.01);
        t += step_size;
        prev_dist = dist;
        
        // Early termination if we've gone too far
        if (t > SDF_MAX_DISTANCE) {
            break;
        }
    }
    
    return result;
}

struct StackEntry {
    node_index: u32,
    t_entry: f32,
    min: vec3<f32>,
    size: f32,
};

// A corrected and optimized stack-based raycast implementation.
fn raycast_octree_stack(ray_origin: vec3<f32>, ray_dir: vec3<f32>, grid_size: u32) -> RayCast {
    var result: RayCast;
    result.pos = vec3<f32>(-1.0);
    result.data = 0u;
    result.steps = 0u;

    let gs_f = f32(grid_size);
    let root_min = vec3<f32>(0.0);
    let root_max = vec3<f32>(gs_f);

    let root_tt = ray_aabb_intersect(ray_origin, ray_dir, root_min, root_max);
    if (root_tt.x > root_tt.y) {
        return result;
    }

    var stack: array<StackEntry, MAX_STACK>;
    var sp: u32 = 0u;
    var intersection_tests: u32 = 1u; // Count the root intersection test

    // Push the root node onto the stack.
    stack[0] = StackEntry(0u, max(root_tt.x, 0.0), root_min, gs_f);
    sp = 1u;

    var child_octant_list: array<u32, 8>;
    var child_t_list: array<f32, 8>;

    while (sp > 0u && intersection_tests < MAX_INTERSECTION_TESTS) {
        sp = sp - 1u;
        let entry = stack[sp];
        result.steps = result.steps + 1u;

        let node = nodes[entry.node_index];
        let is_leaf = (node.firstChildOrData & LEAF_BIT) != 0u;

        if (is_leaf || entry.size <= 1.0) {
            let data = node.firstChildOrData & ~LEAF_BIT;
            if (data > 0u) {
                let t_hit = max(entry.t_entry, 0.0);
                result.pos = ray_origin + ray_dir * t_hit;
                result.data = data;

                let leaf_center = entry.min + vec3<f32>(entry.size * 0.5);
                let p = result.pos - leaf_center;
                let abs_p = abs(p);
                let is_x = f32(abs_p.x > abs_p.y && abs_p.x > abs_p.z);
                let is_y = f32(abs_p.y > abs_p.x && abs_p.y > abs_p.z);
                let is_z = 1.0 - is_x - is_y;
                result.normal = sign(p) * vec3<f32>(is_x, is_y, is_z);
                return result;
            }
            continue;
        }

        let child_mask = node.childMask;
        if (child_mask == 0u) {
            continue;
        }

        let child_size = entry.size * 0.5;
        var child_count = 0u;

        for (var oct: u32 = 0u; oct < 8u; oct = oct + 1u) {
            if ((child_mask & (1u << oct)) != 0u) {
                let cmin = child_min_from_parent(entry.min, entry.size, oct);
                let cmax = cmin + vec3<f32>(child_size);
                let tt = ray_aabb_intersect(ray_origin, ray_dir, cmin, cmax);
                intersection_tests = intersection_tests + 1u;

                if (tt.x <= tt.y && tt.y >= entry.t_entry) {
                    child_octant_list[child_count] = oct;
                    child_t_list[child_count] = max(tt.x, entry.t_entry);
                    child_count = child_count + 1u;
                }
            }
        }

        if (child_count == 0u) { continue; }

        // Sort the intersected children by their t-value (insertion sort)
        for (var i: u32 = 1u; i < child_count; i = i + 1u) {
            let key_oct = child_octant_list[i];
            let key_t = child_t_list[i];
            var j = i;
            while (j > 0u && child_t_list[j - 1u] > key_t) {
                child_octant_list[j] = child_octant_list[j - 1u];
                child_t_list[j] = child_t_list[j - 1u];
                j = j - 1u;
            }
            child_octant_list[j] = key_oct;
            child_t_list[j] = key_t;
        }

        // Push sorted children onto the main stack in reverse order
        for (var i: u32 = 0u; i < child_count; i = i + 1u) {
            let sorted_index = child_count - 1u - i;
            let octant = child_octant_list[sorted_index];

            if (sp >= MAX_STACK) { return result; } // Stack overflow

            let child_offset = countOneBits(child_mask & ((1u << octant) - 1u));
            let child_node_index = node.firstChildOrData + child_offset;
            let cmin = child_min_from_parent(entry.min, entry.size, octant);

            stack[sp] = StackEntry(child_node_index, child_t_list[sorted_index], cmin, child_size);
            sp = sp + 1u;
        }
    }

    return result;
}

struct FragmentOutput {
  @location(0) colorForCanvas: vec4<f32>,
  @location(1) colorForTexture: vec4<f32>,
  @location(2) worldPosition: vec4<f32>,
  @location(3) heatmap: vec4<f32>,
};

// Convert step count to heatmap color
fn steps_to_heatmap_color(steps: u32, max_steps: u32) -> vec3<f32> {
    let x = clamp(f32(steps) / f32(max_steps), 0.0, 1.0);

    // Gradient using smoothstep for nicer transitions
    let c1 = vec3(0.0, 0.0, 1.0); // Blue
    let c2 = vec3(0.0, 1.0, 0.0); // Green
    let c3 = vec3(1.0, 1.0, 0.0); // Yellow
    let c4 = vec3(1.0, 0.0, 0.0); // Red

    let t1 = smoothstep(0.0, 0.4, x);
    let t2 = smoothstep(0.4, 0.7, x);
    let t3 = smoothstep(0.7, 1.0, x);

    var color = mix(c1, c2, t1);
    color = mix(color, c3, t2);
    color = mix(color, c4, t3);

    return color;
};

// Calculate motion vectors for TAA
fn calculate_motion_vector(world_pos: vec3<f32>, current_uv: vec2<f32>, ray_dir: vec3<f32>, camera_pos: vec3<f32>) -> vec2<f32> {
    let has_hit = world_pos.x >= -0.001;
    let far_distance = 10000.0;
    let pos_for_motion = select(ray_dir * far_distance, world_pos, has_hit);

    // Create non-jittered projection by removing jitter from the current perspective matrix
    var unjittered_perspective = context.perspective;
    unjittered_perspective[2][0] -= context.jitter_offset.x * 2.0; // Remove jitter from x offset
    unjittered_perspective[2][1] -= context.jitter_offset.y * 2.0; // Remove jitter from y offset
    
    // Transform world position to current frame's clip space (without jitter)
    let current_view_proj = unjittered_perspective * context.view;
    let current_clip = current_view_proj * vec4<f32>(pos_for_motion, 1.0);
    let current_ndc = current_clip.xyz / current_clip.w;
    let current_screen_uv = vec2<f32>(current_ndc.x * 0.5 + 0.5, -current_ndc.y * 0.5 + 0.5);
    
    // Transform world position to previous frame's clip space (already non-jittered)
    let prev_clip = context.prev_view_projection * vec4<f32>(pos_for_motion, 1.0);
    let prev_ndc = prev_clip.xyz / prev_clip.w;
    let prev_screen_uv = vec2<f32>(prev_ndc.x * 0.5 + 0.5, -prev_ndc.y * 0.5 + 0.5);
    
    let motion_vec = prev_screen_uv - current_screen_uv;

    // For background, filter out small movements (translation)
    let motion_magnitude = length(motion_vec);
    let is_small_motion = motion_magnitude < 0.001;
    
    return select(motion_vec, vec2<f32>(0.0), !has_hit && is_small_motion);
}

// A more robust and stable TAA implementation.
// A more robust TAA that handles disocclusion edges gracefully.
fn taa_sample_history(current_uv: vec2<f32>, current_color: vec4<f32>, world_hit_pos: vec3<f32>, camera_pos: vec3<f32>, ray_dir: vec3<f32>) -> vec4<f32> {
    let has_current_hit = world_hit_pos.x >= -0.001;
    
    // 1. Calculate motion vector to find the sample location in the previous frame.
    let motion_vector = calculate_motion_vector(world_hit_pos, current_uv, ray_dir, camera_pos);
    let history_uv = current_uv + motion_vector;

    // FIX: Perform ALL texture samples/loads upfront, in uniform control flow.
    let history_coord_i = vec2<i32>(clamp(history_uv, vec2(0.0), vec2(0.999)) * context.resolution);
    let prev_frame_data = textureLoad(prevWorldPosTexture, history_coord_i, 0);
    let history_color = textureSample(prevFrameTexture, smpler, history_uv);
    
    // Sample current pixel neighborhood for voxel edge detection and color bounds
    let pixel_coord = vec2<i32>(current_uv * context.resolution);
    
    // Unrolled neighborhood check
    let n0 = textureLoad(prevWorldPosTexture, pixel_coord + vec2(-1, -1), 0).w > 0.0;
    let n1 = textureLoad(prevWorldPosTexture, pixel_coord + vec2( 0, -1), 0).w > 0.0;
    let n2 = textureLoad(prevWorldPosTexture, pixel_coord + vec2( 1, -1), 0).w > 0.0;
    let n3 = textureLoad(prevWorldPosTexture, pixel_coord + vec2(-1,  0), 0).w > 0.0;
    let n4 = textureLoad(prevWorldPosTexture, pixel_coord + vec2( 1,  0), 0).w > 0.0;
    let n5 = textureLoad(prevWorldPosTexture, pixel_coord + vec2(-1,  1), 0).w > 0.0;
    let n6 = textureLoad(prevWorldPosTexture, pixel_coord + vec2( 0,  1), 0).w > 0.0;
    let n7 = textureLoad(prevWorldPosTexture, pixel_coord + vec2( 1,  1), 0).w > 0.0;
    let n8 = textureLoad(prevWorldPosTexture, pixel_coord, 0).w > 0.0;

    let near_voxel_edge = n0 || n1 || n2 || n3 || n4 || n5 || n6 || n7 || n8;
    
    // For background pixels, check if we're near voxel edges for anti-aliasing
    if (!has_current_hit && !near_voxel_edge) {
        // Background pixel not near voxel edges, skip TAA history entirely
        return current_color;
    }

    // 2. Perform History Rejection Checks.
    let is_in_bounds = all(history_uv >= vec2<f32>(0.0)) && all(history_uv <= vec2<f32>(1.0));
    let prev_data_valid = abs(prev_frame_data.w) > 0.5;
    let had_prev_hit = prev_frame_data.w > 0.0;
    var is_history_valid = is_in_bounds && prev_data_valid && (has_current_hit == had_prev_hit);

    if (is_history_valid && has_current_hit) {
        let prev_world_pos = prev_frame_data.xyz;
        let world_pos_diff = distance(world_hit_pos, prev_world_pos);
        let depth = distance(world_hit_pos, camera_pos);
        let rejection_threshold = max(0.5, depth * 0.01);
        if (world_pos_diff > rejection_threshold) {
            is_history_valid = false;
        }
    }

    // 4. If history is invalid, we clamp the history color to current neighborhood.
    let current_min = current_color * 0.8;  // Allow some variation
    let current_max = current_color * 1.2;
    let clamped_history = clamp(history_color, current_min, current_max);
    let blend_target = select(clamped_history, current_color, is_history_valid);

    // 5. Blend with adaptive blend factor based on history validity and motion
    let motion_magnitude = length(motion_vector);
    let valid_history_blend = mix(0.01, 0.1, clamp(motion_magnitude * 10.0, 0.0, 1.0));
    let invalid_history_blend = 0.3;
    let blend_factor = select(invalid_history_blend, valid_history_blend, is_history_valid);
    
    return mix(history_color, blend_target, blend_factor);
}

@fragment
fn main_fs(@builtin(position) pos: vec4<f32>) -> FragmentOutput {
  let uv = pos.xy / context.resolution;
  let ndc = vec4<f32>((uv - 0.5) * 2.0 * vec2<f32>(1.0, -1.0), 0.0, 1.0);

  // Transform from NDC to world space to get ray
  let view_pos = context.inverse_perspective * ndc;
  var world_pos = context.inverse_view * vec4<f32>(view_pos.xyz / view_pos.w, 1.0);

  // Camera position in world space
  var camera_pos = (context.inverse_view * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;

  // Create ray direction
  let ray_dir = normalize(world_pos.xyz - camera_pos);

  // Raycast through octree or distance field based on render mode
  var hit: RayCast;
  if (context.render_mode >= 2u) {
    // Use distance field raymarching (modes 2 and 3)
    hit = raycast_distance_field(camera_pos, ray_dir);
  } else {
    // Use octree traversal (modes 0 and 1)
    hit = raycast_octree_stack(camera_pos, ray_dir, context.grid_size);
  }

  var color: vec4<f32>;
  if (hit.pos.x >= -0.001) {
    // Hit something in the octree
      // 💡 Assume your raycaster now also returns the surface normal
      let normal = normalize(hit.normal);

      // --- 1. DEFINE LIGHTING PROPERTIES ---
      let light_pos = vec3<f32>(f32(context.grid_size) * 0.8, f32(context.grid_size) * 1.2, f32(context.grid_size) * 0.3);
      let light_dir = normalize(light_pos - hit.pos);
      let object_color = vec3<f32>(1.0, 1.0, 1.0); // White color
      
      // Sky light properties - use background color
      let sky_color = abs(ray_dir) * 0.5 + 0.5; // Same as background
      let sky_intensity = 0.3;
      let sky_samples = 1u; // Fewer samples for sky light (performance)
      
      // Sun light parameters  
      let light_size = 0.5; // Size of the light source (affects shadow softness)

      // --- 2. CALCULATE DIFFUSE LIGHTING (LAMBERTIAN) ---
      // The max() prevents surfaces facing away from the light from becoming negative
      let diffuse_intensity = max(dot(normal, light_dir), 0.0);

      // --- 3. CALCULATE RAYTRACED SHADOWS ---
      var shadow_factor = 1.0;
      if (diffuse_intensity > 0.0) {
          shadow_factor = calculate_soft_shadow(hit.pos, light_pos, light_size, context.grid_size, uv);
      }

      // --- 4. CALCULATE SKY LIGHT ---
      let sky_visibility = calculate_sky_light(hit.pos, normal, context.grid_size, sky_samples, uv);
      let sky_contribution = sky_color * sky_intensity * sky_visibility;

      // --- 5. (OPTIONAL) ADD SPECULAR HIGHLIGHTS (BLINN-PHONG) ---
      let view_dir = normalize(camera_pos - hit.pos);
      let half_dir = normalize(light_dir + view_dir);
      let shininess = 8.0; // Lower value = larger, softer highlight (more matte)
      var specular_intensity = pow(max(dot(normal, half_dir), 0.0), shininess) * 0.3; // Reduce intensity
      // Only add specular light if there is diffuse light and no shadow
      if (diffuse_intensity <= 0.0 || shadow_factor <= 0.0) {
        specular_intensity = 0.0;
      }

      // --- 6. COMBINE LIGHTING WITH SHADOWS AND SKY ---
      // Sun light (directional with shadows)
      let sun_diffuse = diffuse_intensity * shadow_factor;
      let sun_specular = specular_intensity * shadow_factor;
      let sun_contribution = object_color * sun_diffuse + vec3<f32>(1.0) * sun_specular;
      
      // Final lighting = sky light + sun light
      let lit_color = object_color * sky_contribution + sun_contribution;


      color = vec4(lit_color, 1);

  } else {
    // Debug: Show ray direction as color
    color = vec4<f32>(abs(ray_dir) * 0.5 + 0.5, 1.0);
  }

  // Mouse interaction
//  var ar = context.resolution.x / context.resolution.y;
//  var p = (uv - 0.5) * 2.0;
//  p.x *= ar;
//  var mouse = context.mouse_rel;
//  mouse.x *= ar;
//  let mouse_dist = distance(p, mouse);
//  if (mouse_dist < 0.5) {
//	color = vec4<f32>(hit.normal * 0.5 + 0.5, 1);
//  }

  // TAA Implementation - pass the world position from our raycast (only if enabled)
  let final_color = select(color, taa_sample_history(uv, color, hit.pos, camera_pos, ray_dir), context.taa_enabled != 0u);
  
  // Generate heatmap color based on step count
  let max_expected_steps = 48u;  // Lower value for more visible heatmap differences
  let heatmap_color = steps_to_heatmap_color(hit.steps, max_expected_steps);
  
  var output: FragmentOutput;
  
  // Choose output based on render mode
  if (context.render_mode == 0u || context.render_mode == 2u) {
    // Normal rendering mode (octree or SDF)
    output.colorForTexture = final_color;
    output.colorForCanvas = final_color;
    output.heatmap = vec4(0.0, 0.0, 0.0, 1.0); // Black heatmap when not in heatmap mode
  } else {
    // Heatmap rendering mode (octree or SDF)
    let heatmap_output = vec4(heatmap_color, 1.0);
    output.colorForTexture = heatmap_output;
    output.colorForCanvas = heatmap_output;
    output.heatmap = heatmap_output;
  }
  
  // Store world position for next frame's TAA (w component stores hit validity)
  output.worldPosition = vec4<f32>(hit.pos, select(-1.0, 1.0, hit.pos.x >= -0.001));
  return output;
}
