#import "../data/context.wgsl"

struct Octree {
    data: u32,
    childs: array<u32, 8>
}

const INVALID_INDEX: u32 = 0xFFFFFFFFu;
const MAX_STACK: u32 = 24u; // must be >= context.max_depth + a small margin

@group(0) @binding(0) var <uniform> context: Context;
@group(0) @binding(1) var<storage, read_write> nodes: array<Octree>;
@group(1) @binding(0) var prevFrameTexture: texture_2d<f32>;
@group(1) @binding(1) var smpler: sampler;
@group(1) @binding(2) var prevWorldPosTexture: texture_2d<f32>;

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
    var cm = parent_min;
    // x
    if ((octant & 1u) != 0u) { cm.x = cm.x + half; }
    // y
    if ((octant & 2u) != 0u) { cm.y = cm.y + half; }
    // z
    if ((octant & 4u) != 0u) { cm.z = cm.z + half; }
    return cm;
}

struct RayCast {
	pos: vec3<f32>,
	normal: vec3<f32>,
	data: u32,
	steps: u32
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
fn random_cone_direction(seed: f32, cone_angle: f32, light_dir: vec3<f32>) -> vec3<f32> {
    // Simple pseudo-random based on seed
    let a = fract(sin(seed * 12.9898) * 43758.5453);
    let b = fract(sin(seed * 93.9898) * 28001.8384);
    
    // Convert to spherical coordinates within cone
    let theta = a * 2.0 * 3.14159;
    let phi = acos(1.0 - b * (1.0 - cos(cone_angle)));
    
    // Create orthonormal basis around light direction
    let w = light_dir;
    var cross_vec = vec3(0.0, 1.0, 0.0);
    if (abs(w.x) <= 0.1) {
        cross_vec = vec3(1.0, 0.0, 0.0);
    }
    let u = normalize(cross(cross_vec, w));
    let v = cross(w, u);
    
    // Convert spherical to cartesian in local basis
    let sin_phi = sin(phi);
    let local_dir = vec3(sin_phi * cos(theta), sin_phi * sin(theta), cos(phi));
    
    // Transform to world space
    return normalize(u * local_dir.x + v * local_dir.y + w * local_dir.z);
}

// Calculate soft shadows by sampling multiple rays within light area
fn calculate_soft_shadow(hit_pos: vec3<f32>, light_pos: vec3<f32>, light_size: f32, grid_size: u32, samples: u32) -> f32 {
    var shadow_factor = 0.0;
    let base_seed = dot(hit_pos, vec3(12.9898, 78.233, 37.719));
    
    for (var i: u32 = 0u; i < samples; i++) {
        let seed = base_seed + f32(i) * 7.531;
        
        // Generate random point on light area
        let light_dir = normalize(light_pos - hit_pos);
        let random_dir = random_cone_direction(seed, light_size, light_dir);
        let sample_light_pos = light_pos + random_dir * light_size * 50.0; // Scale factor for light area
        
        if (!cast_shadow_ray(hit_pos, sample_light_pos, grid_size)) {
            shadow_factor += 1.0;
        }
    }
    
    return shadow_factor / f32(samples);
}

// Generate random direction in hemisphere above surface normal
fn random_hemisphere_direction(seed: f32, normal: vec3<f32>) -> vec3<f32> {
    // Simple pseudo-random based on seed
    let a = fract(sin(seed * 12.9898) * 43758.5453);
    let b = fract(sin(seed * 93.9898) * 28001.8384);
    
    // Generate uniform distribution on hemisphere
    let theta = a * 2.0 * 3.14159;
    let phi = acos(sqrt(b)); // Cosine-weighted distribution
    
    // Create orthonormal basis around normal
    let w = normal;
    var cross_vec = vec3(0.0, 1.0, 0.0);
    if (abs(w.y) > 0.9) {
        cross_vec = vec3(1.0, 0.0, 0.0);
    }
    let u = normalize(cross(cross_vec, w));
    let v = cross(w, u);
    
    // Convert spherical to cartesian in local basis
    let sin_phi = sin(phi);
    let local_dir = vec3(sin_phi * cos(theta), sin_phi * sin(theta), cos(phi));
    
    // Transform to world space
    return normalize(u * local_dir.x + v * local_dir.y + w * local_dir.z);
}

// Calculate sky light contribution by sampling hemisphere
fn calculate_sky_light(hit_pos: vec3<f32>, normal: vec3<f32>, grid_size: u32, samples: u32) -> f32 {
    var sky_factor = 0.0;
    let base_seed = dot(hit_pos, vec3(17.531, 41.234, 63.719));
    
    for (var i: u32 = 0u; i < samples; i++) {
        let seed = base_seed + f32(i) * 9.234;
        let sky_dir = random_hemisphere_direction(seed, normal);
        
        // Cast ray towards sky - if it doesn't hit anything, we see sky
        let sky_hit = raycast_octree_stack(hit_pos + normal * 0.001, sky_dir, grid_size);
        if (sky_hit.pos.x < -0.001) {
            // Ray escaped to sky
            sky_factor += 1.0;
        }
    }
    
    return sky_factor / f32(samples);
}

// Stack-based octree raycast. Returns world-space hit position or (-1,-1,-1)
fn raycast_octree_stack(ray_origin: vec3<f32>, ray_dir: vec3<f32>, grid_size: u32) -> RayCast {
	var result: RayCast;
    // Root AABB in world coordinates: [0, grid_size]
    let gs_f = f32(grid_size);
    let root_min = vec3<f32>(0.0, 0.0, 0.0);
    let root_max = vec3<f32>(gs_f, gs_f, gs_f);

    // quick root intersection
    let root_tt = ray_aabb_intersect(ray_origin, ray_dir, root_min, root_max);
    if (root_tt.x > root_tt.y) {
    	// no hit
    	result.data = 0;
    	result.pos = vec3<f32>(-1.0);
    	return result;
    }

    // stack arrays (LIFO)
    var stack_node: array<u32, MAX_STACK>;
    var stack_min: array<vec3<f32>, MAX_STACK>;
    var stack_size: array<f32, MAX_STACK>;
    var stack_tentry: array<f32, MAX_STACK>;
    var stack_depth_int: array<u32, MAX_STACK>;
    var sp: u32 = 0u;

    // push root
    stack_node[0] = 0u;
    stack_min[0] = root_min;
    stack_size[0] = gs_f;
    stack_tentry[0] = max(root_tt.x, 0.0);
    stack_depth_int[0] = 0u;
    sp = 1u;

    // temp per-node child list (max 8)
    var child_idx_list: array<u32, 8>;
    var child_t_list: array<f32, 8>;
    var child_count: u32;

    while (sp > 0u) {
        // pop
        sp = sp - 1u;
        let node_index = stack_node[sp];
        let node_min = stack_min[sp];
        let node_size_f = stack_size[sp];
        let node_tentry = stack_tentry[sp];
        let depth = stack_depth_int[sp];

        // determine leaf: either reached provided max depth or voxel-size <= 1.0 (world-space units)
        if ((depth >= context.max_depth) || (node_size_f <= 1.0)) {
            // It's a leaf level — check data payload
            let d = nodes[node_index].data;
            if (d > 0u) {
                // return approximate hit position using node_tentry (first intersection with leaf AABB)
                let t_hit = max(node_tentry, 0.0);
                result.pos = ray_origin + ray_dir * t_hit;
                result.data = d;

                // --- 👇 NORMAL CALCULATION LOGIC ---
				// Calculate the center of the leaf cube
				let leaf_center = node_min + vec3<f32>(node_size_f * 0.5);
				// Vector from the center to the hit point
				let p = result.pos - leaf_center;
				// Find the axis with the largest component
				let abs_p = abs(p);
				if (abs_p.x > abs_p.y && abs_p.x > abs_p.z) {
					result.normal = vec3<f32>(sign(p.x), 0.0, 0.0);
				} else if (abs_p.y > abs_p.z) {
					result.normal = vec3<f32>(0.0, sign(p.y), 0.0);
				} else {
					result.normal = vec3<f32>(0.0, 0.0, sign(p.z));
				}
				// --- 👆 END OF NORMAL CALCULATION ---

                return result;
            }
            // else no payload -> continue
            continue;
        }

        // internal node: test 8 children for intersection
        child_count = 0u;
        let child_size = node_size_f * 0.5;

        // test all 8 octants
        for (var oct: u32 = 0u; oct < 8u; oct = oct + 1u) {
            // compute child AABB
            let cmin = child_min_from_parent(node_min, node_size_f, oct);
            let cmax = cmin + vec3<f32>(child_size, child_size, child_size);

            let tt = ray_aabb_intersect(ray_origin, ray_dir, cmin, cmax);

            // The intersection interval [tt.x, tt.y] must be valid AND
            // overlap with the parent's interval, which starts at node_tentry.
            if (tt.x <= tt.y && tt.y >= node_tentry) {
                // check that child exists
                let child_node_index = nodes[node_index].childs[oct];
                if (child_node_index != INVALID_INDEX) {
                    child_idx_list[child_count] = oct;

                    // The child's real entry point is the LATER of its own entry
                    // or its parent's entry. This enforces monotonic t-progression.
                    child_t_list[child_count] = max(tt.x, node_tentry);
                    child_count = child_count + 1u;
                }
            }
        }

        if (child_count == 0u) {
            // nothing below this node
            continue;
        }

        // sort child lists by entry t ascending (insertion sort for <= 8 elements)
        for (var i: u32 = 1u; i < child_count; i = i + 1u) {
            let key_oct = child_idx_list[i];
            let key_t = child_t_list[i];
            var j: u32 = i;
            loop {
                if ((j == 0u) || (child_t_list[j - 1u] <= key_t)) { break; }
                child_idx_list[j] = child_idx_list[j - 1u];
                child_t_list[j] = child_t_list[j - 1u];
                j = j - 1u;
            }
            child_idx_list[j] = key_oct;
            child_t_list[j] = key_t;
        }

        // push children in reverse sorted order so nearest is popped first
        for (var k: i32 = i32(child_count) - 1; k >= 0; k = k - 1) {
            let oct = child_idx_list[u32(k)];
            let cmin = child_min_from_parent(node_min, node_size_f, oct);
            let child_node_index = nodes[node_index].childs[oct];
            if (child_node_index == INVALID_INDEX) {
                continue;
            }

            // push
            if (sp >= MAX_STACK) {
                // stack overflow guard — bail out
                result.data = 0;
                result.pos = vec3<f32>(-1.0);
                return result;
            }
            stack_node[sp] = child_node_index;
            stack_min[sp] = cmin;
            stack_size[sp] = child_size;
            stack_tentry[sp] = child_t_list[u32(k)];
            stack_depth_int[sp] = depth + 1u;
            sp = sp + 1u;
        }
    }

    // nothing hit
	result.data = 0;
	result.pos = vec3<f32>(-1.0);
    return result;
}

struct FragmentOutput {
  @location(0) colorForCanvas: vec4<f32>,
  @location(1) colorForTexture: vec4<f32>,
  @location(2) worldPosition: vec4<f32>,
};

// Calculate motion vectors for TAA
fn calculate_motion_vector(world_pos: vec3<f32>, current_uv: vec2<f32>) -> vec2<f32> {
    // Transform world position to current frame's clip space (for verification)
    let current_view_proj = context.perspective * context.view;
    let current_clip = current_view_proj * vec4<f32>(world_pos, 1.0);
    let current_ndc = current_clip.xyz / current_clip.w;
    let current_screen_uv = current_ndc.xy * 0.5 + 0.5;
    // Note: Y is already flipped correctly by the projection matrix
    
    // Transform world position to previous frame's clip space
    let prev_clip = context.prev_view_projection * vec4<f32>(world_pos, 1.0);
    let prev_ndc = prev_clip.xyz / prev_clip.w;
    let prev_screen_uv = prev_ndc.xy * 0.5 + 0.5;
    
    // Motion vector: where to find this pixel in the previous frame
    // This is the offset from current position to previous position
    return prev_screen_uv - current_screen_uv;
}

// TAA history sampling with validation - optimized to use stored world position
fn taa_sample_history(current_uv: vec2<f32>, current_color: vec4<f32>, world_hit_pos: vec3<f32>) -> vec4<f32> {
    // Use the world position from the main raycast (passed as parameter)
    // No need to raycast again!
    
    // Calculate motion vector if we have a valid hit
    let has_hit = select(0.0, 1.0, world_hit_pos.x >= -0.001);
    let motion_vector = calculate_motion_vector(world_hit_pos, current_uv) * has_hit;
    let history_uv = current_uv + motion_vector;
    
    // Much stricter motion vector validation to reduce smearing
    let mv_length = length(motion_vector);
    let mv_reasonable = select(0.0, 1.0, mv_length < 0.05); // Much stricter threshold
    
    // Strict bounds checking with margin
    let in_bounds = select(0.0, 1.0, 
        history_uv.x >= 0.02 && history_uv.x <= 0.98 && 
        history_uv.y >= 0.02 && history_uv.y <= 0.98);
    
    let history_valid = has_hit * in_bounds * mv_reasonable;
    
    // Sample history with improved filtering
    let history_color = textureSample(prevFrameTexture, smpler, history_uv);
    
    // Expand neighborhood sampling for better color variance analysis
    let coord = vec2<i32>(current_uv * context.resolution);
    let center = textureLoad(prevFrameTexture, coord, 0);
    let tl = textureLoad(prevFrameTexture, coord + vec2<i32>(-1, -1), 0);
    let tr = textureLoad(prevFrameTexture, coord + vec2<i32>(1, -1), 0);
    let bl = textureLoad(prevFrameTexture, coord + vec2<i32>(-1, 1), 0);
    let br = textureLoad(prevFrameTexture, coord + vec2<i32>(1, 1), 0);
    let left = textureLoad(prevFrameTexture, coord + vec2<i32>(-1, 0), 0);
    let right = textureLoad(prevFrameTexture, coord + vec2<i32>(1, 0), 0);
    let top = textureLoad(prevFrameTexture, coord + vec2<i32>(0, -1), 0);
    let bottom = textureLoad(prevFrameTexture, coord + vec2<i32>(0, 1), 0);
    
    // Calculate neighborhood statistics with better coverage
    let neighbor_min = min(min(min(min(min(min(min(min(tl, tr), bl), br), left), right), top), bottom), center);
    let neighbor_max = max(max(max(max(max(max(max(max(tl, tr), bl), br), left), right), top), bottom), center);
    let neighbor_avg = (tl + tr + bl + br + left + right + top + bottom + center) / 9.0;
    
    // Calculate color variance in neighborhood for stability assessment
    let color_variance = (
        length(tl - neighbor_avg) + length(tr - neighbor_avg) + 
        length(bl - neighbor_avg) + length(br - neighbor_avg) +
        length(left - neighbor_avg) + length(right - neighbor_avg) +
        length(top - neighbor_avg) + length(bottom - neighbor_avg) +
        length(center - neighbor_avg)
    ) / 9.0;
    
    // Variance-based stability: high variance areas need more temporal stability
    let stability_threshold = 0.12;
    let is_stable_area = select(0.0, 1.0, color_variance < stability_threshold);
    
    // Use expanded neighborhood clamping for better temporal coherence
    let clamp_margin = 0.015;
    let expanded_neighbor_min = neighbor_min - vec4<f32>(clamp_margin);
    let expanded_neighbor_max = neighbor_max + vec4<f32>(clamp_margin);
    let clamped_history = clamp(history_color, expanded_neighbor_min, expanded_neighbor_max);
    
    // Enhanced luminance and color difference validation
    let current_luma = dot(current_color.rgb, vec3<f32>(0.299, 0.587, 0.114));
    let history_luma = dot(clamped_history.rgb, vec3<f32>(0.299, 0.587, 0.114));
    let luma_diff = abs(current_luma - history_luma);
    let color_diff = length(current_color.rgb - clamped_history.rgb);
    
    // Multi-factor temporal confidence metric with luminance-adaptive thresholds
    let camera_speed = length(context.camera_velocity);
    let motion_stability = select(0.3, 1.0, camera_speed < 2.0); // Reduce confidence during fast camera motion
    
    // Adaptive thresholds based on luminance - be more permissive in dark areas
    let avg_luma = (current_luma + history_luma) * 0.5;
    let dark_area_factor = select(1.0, 2.5, avg_luma < 0.15); // More permissive in dark areas
    
    let adaptive_luma_threshold = 0.12 * dark_area_factor;
    let adaptive_color_threshold = 0.15 * dark_area_factor;
    
    let luma_confidence = select(0.0, 1.0, luma_diff < adaptive_luma_threshold);
    let color_confidence = select(0.0, 1.0, color_diff < adaptive_color_threshold);
    let temporal_confidence = history_valid * luma_confidence * color_confidence * is_stable_area * motion_stability;
    
    // Adaptive blending with luminance-aware anti-smearing bias
    let base_conservative_blend = 0.06;
    let base_moderate_blend = 0.22;        
    let base_aggressive_blend = 0.75;
    
    // Be more conservative in dark areas to preserve indirect lighting accumulation
    let dark_conservation_factor = select(1.0, 0.4, avg_luma < 0.2); // Much more conservative in dark areas
    
    let conservative_blend = base_conservative_blend * dark_conservation_factor;
    let moderate_blend = base_moderate_blend * dark_conservation_factor;
    let aggressive_blend = mix(base_aggressive_blend, base_moderate_blend, 1.0 - dark_conservation_factor);
    
    let blend_factor = select(
        aggressive_blend,
        select(moderate_blend, conservative_blend, temporal_confidence > 0.75),
        temporal_confidence < 0.25
    );
    
    // Final temporal blend with improved fallback strategy
    let fallback_color = mix(center, current_color, 0.35);
    let selected_history = select(fallback_color, clamped_history, history_valid > 0.5);
    
    return mix(selected_history, current_color, blend_factor);
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
          shadow_factor = calculate_soft_shadow(hit.pos, light_pos, light_size, context.grid_size, shadow_samples);
      }

      // --- 4. CALCULATE SKY LIGHT ---
      let sky_visibility = calculate_sky_light(hit.pos, normal, context.grid_size, sky_samples);
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
  let history_color = taa_sample_history(uv, color, hit.pos);
  
  var output: FragmentOutput;
  output.colorForTexture = history_color;
  output.colorForCanvas = history_color;
  // Store world position for next frame's TAA (w component stores hit validity)
  output.worldPosition = vec4<f32>(hit.pos, select(-1.0, 1.0, hit.pos.x >= -0.001));
  return output;
}
