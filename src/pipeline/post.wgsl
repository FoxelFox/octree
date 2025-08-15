#import "../data/context.wgsl"

struct CompactNode {
    firstChildOrData: u32,
    childMask: u32,
}

const LEAF_BIT: u32 = 0x80000000u;
const INVALID_INDEX: u32 = 0xFFFFFFFFu;
const MAX_STACK: u32 = 24u; // must be >= context.max_depth + a small margin
const MAX_INTERSECTION_TESTS: u32 = 384; // maximum ray-AABB tests before early termination

@group(0) @binding(0) var <uniform> context: Context;
@group(0) @binding(1) var<storage, read> nodes: array<CompactNode>;
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
    
    let shadow_hit = raycast_octree_stack(shadow_start, shadow_dir, grid_size);
    
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
fn calculate_soft_shadow(hit_pos: vec3<f32>, light_pos: vec3<f32>, light_size: f32, grid_size: u32, samples: u32, pixel_uv: vec2<f32>) -> f32 {
    var shadow_factor = 0.0;
    let base_seed = dot(hit_pos, vec3(12.9898, 78.233, 37.719)) + context.random_seed;
    
    for (var i: u32 = 0u; i < samples; i++) {
        let seed = base_seed + f32(i) * 7.531;
        
        // Generate random point on light area
        let light_dir = normalize(light_pos - hit_pos);
        let random_dir = random_cone_direction(seed, light_size, light_dir, pixel_uv);
        let sample_light_pos = light_pos + random_dir * light_size * 50.0; // Scale factor for light area
        
        if (!cast_shadow_ray(hit_pos, sample_light_pos, grid_size)) {
            shadow_factor += 1.0;
        }
    }
    
    return shadow_factor / f32(samples);
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
        
        // Cast ray towards sky - if it doesn't hit anything, we see sky
        let sky_hit = raycast_octree_stack(hit_pos + normal * 0.001, sky_dir, grid_size);
        if (sky_hit.pos.x < -0.001) {
            // Ray escaped to sky
            sky_factor += 1.0;
        }
    }
    
    return sky_factor / f32(samples);
}

// Stack-based octree raycast for the compact node structure.
fn raycast_octree_stack(ray_origin: vec3<f32>, ray_dir: vec3<f32>, grid_size: u32) -> RayCast {
	var result: RayCast;
	result.pos = vec3<f32>(-1.0);
	result.data = 0u;
	result.steps = 0u;

	var intersection_test_count: u32 = 0u;
    let gs_f = f32(grid_size);
    let root_min = vec3<f32>(0.0);
    let root_max = vec3<f32>(gs_f);

    let root_tt = ray_aabb_intersect(ray_origin, ray_dir, root_min, root_max);
    intersection_test_count = intersection_test_count + 1u;
    if (root_tt.x > root_tt.y) {
    	return result;
    }

    var stack_node_index: array<u32, MAX_STACK>;
    var stack_node_min: array<vec3<f32>, MAX_STACK>;
    var stack_node_size: array<f32, MAX_STACK>;
    var stack_tentry: array<f32, MAX_STACK>;
    var sp: u32 = 0u;

    stack_node_index[0] = 0u;
    stack_node_min[0] = root_min;
    stack_node_size[0] = gs_f;
    stack_tentry[0] = max(root_tt.x, 0.0);
    sp = 1u;

    var child_octant_list: array<u32, 8>;
    var child_t_list: array<f32, 8>;

    while (sp > 0u) {
        sp = sp - 1u;
        let node_index = stack_node_index[sp];
        let node_min = stack_node_min[sp];
        let node_size = stack_node_size[sp];
        let node_tentry = stack_tentry[sp];
        
        result.steps = result.steps + 1u;

		let node = nodes[node_index];
		let is_leaf = (node.firstChildOrData & LEAF_BIT) != 0u;

        if (is_leaf || node_size <= 1.0) {
            let data = node.firstChildOrData & ~LEAF_BIT;
            if (data > 0u) {
                let t_hit = max(node_tentry, 0.0);
                result.pos = ray_origin + ray_dir * t_hit;
                result.data = data;

                let leaf_center = node_min + vec3<f32>(node_size * 0.5);
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

        let child_size = node_size * 0.5;
		var child_count = 0u;

        for (var oct: u32 = 0u; oct < 8u; oct = oct + 1u) {
            if ((child_mask & (1u << oct)) != 0u) {
				if (intersection_test_count >= MAX_INTERSECTION_TESTS) { break; }
				
                let cmin = child_min_from_parent(node_min, node_size, oct);
                let cmax = cmin + vec3<f32>(child_size);
                let tt = ray_aabb_intersect(ray_origin, ray_dir, cmin, cmax);
                intersection_test_count = intersection_test_count + 1u;

                if (tt.x <= tt.y && tt.y >= node_tentry) {
                    child_octant_list[child_count] = oct;
                    child_t_list[child_count] = max(tt.x, node_tentry);
                    child_count = child_count + 1u;
                }
            }
        }

		if (intersection_test_count >= MAX_INTERSECTION_TESTS) { break; }
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

            stack_node_index[sp] = child_node_index;
            stack_node_min[sp] = child_min_from_parent(node_min, node_size, octant);
            stack_node_size[sp] = child_size;
            stack_tentry[sp] = child_t_list[sorted_index];
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

  // Raycast through octree (world coords in [0, grid_size])
  let hit = raycast_octree_stack(camera_pos, ray_dir, context.grid_size);

  var color: vec4<f32>;
  if (hit.pos.x >= -0.001) {
    // Hit something in the octree
      // 💡 Assume your raycaster now also returns the surface normal
      let normal = normalize(hit.normal);

      // --- 1. DEFINE LIGHTING PROPERTIES ---
      let light_pos = vec3<f32>(f32(context.grid_size) * 0.8, f32(context.grid_size) * 1.2, f32(context.grid_size) * 0.3);
      let light_dir = normalize(light_pos - hit.pos);
      let object_color = vec3<f32>(0.95, 0.95, 0.90); // Alpina white color
      
      // Sky light properties - use background color
      let sky_color = abs(ray_dir) * 0.5 + 0.5; // Same as background
      let sky_intensity = 0.3;
      let sky_samples = 1u; // Fewer samples for sky light (performance)
      
      // Sun light parameters  
      let light_size = 0.5; // Size of the light source (affects shadow softness)
      let shadow_samples = 1u; // Number of shadow rays (higher = smoother but slower)

      // --- 2. CALCULATE DIFFUSE LIGHTING (LAMBERTIAN) ---
      // The max() prevents surfaces facing away from the light from becoming negative
      let diffuse_intensity = max(dot(normal, light_dir), 0.0);

      // --- 3. CALCULATE RAYTRACED SHADOWS ---
      var shadow_factor = 1.0;
      if (diffuse_intensity > 0.0) {
          shadow_factor = calculate_soft_shadow(hit.pos, light_pos, light_size, context.grid_size, shadow_samples, uv);
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

  // TAA Implementation - pass the world position from our raycast
  let history_color = taa_sample_history(uv, color, hit.pos, camera_pos, ray_dir);
  
  // Generate heatmap color based on step count
  let max_expected_steps = 48u;  // Lower value for more visible heatmap differences
  let heatmap_color = steps_to_heatmap_color(hit.steps, max_expected_steps);
  
  var output: FragmentOutput;
  
  // Choose output based on render mode
  if (context.render_mode == 0u) {
    // Normal rendering mode
    output.colorForTexture = history_color;
    output.colorForCanvas = history_color;
    output.heatmap = vec4(0.0, 0.0, 0.0, 1.0); // Black heatmap when not in heatmap mode
  } else {
    // Heatmap rendering mode
    let heatmap_output = vec4(heatmap_color, 1.0);
    output.colorForTexture = heatmap_output;
    output.colorForCanvas = heatmap_output;
    output.heatmap = heatmap_output;
  }
  
  // Store world position for next frame's TAA (w component stores hit validity)
  output.worldPosition = vec4<f32>(hit.pos, select(-1.0, 1.0, hit.pos.x >= -0.001));
  return output;
}
