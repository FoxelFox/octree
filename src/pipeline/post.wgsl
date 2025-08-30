#import "../data/context.wgsl"

struct Octree {
    data: u32,
    childs: array<u32, 8>
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

const SDF_MAX_DISTANCE: f32 = 100000.0; // Maximum ray distance
const SDF_OVER_RELAXATION: f32 = 0.8; // Step size multiplier for better precision

@group(0) @binding(0) var <uniform> context: Context;
@group(0) @binding(1) var<storage, read> nodes: array<Octree>;
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
    if (context.render_mode >= 4u) {
        // Use hybrid raymarching
        shadow_hit = raycast_hybrid(shadow_start, shadow_dir, grid_size);
    } else if (context.render_mode >= 2u) {
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
        if (context.render_mode >= 4u) {
            // Use hybrid raymarching
            sky_hit = raycast_hybrid(hit_pos + normal * 0.001, sky_dir, grid_size);
        } else if (context.render_mode >= 2u) {
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

// Trilinear interpolated distance field sampling for smooth surfaces
fn sample_distance_field(pos: vec3<f32>) -> f32 {
    let gs_f = f32(context.grid_size);
    
    // Early bounds check with single comparison
    let in_bounds = all(pos >= vec3<f32>(0.0)) && all(pos < vec3<f32>(gs_f - 1.0));
    if (!in_bounds) {
        return 1.0; // Outside grid = empty space
    }
    
    // Get the grid position and fractional part for interpolation
    let grid_pos = clamp(pos, vec3<f32>(0.0), vec3<f32>(gs_f - 1.001));
    let base_pos = floor(grid_pos);
    let frac = grid_pos - base_pos;
    
    // Pre-calculate grid size and base index
    let gs = context.grid_size;
    let base_x = u32(base_pos.x);
    let base_y = u32(base_pos.y); 
    let base_z = u32(base_pos.z);
    
    // Sample 8 corner voxels for trilinear interpolation
    // Organize as a 2x2x2 cube: [z][y][x]
    let gs_sq = gs * gs;
    let base_idx = base_z * gs_sq + base_y * gs + base_x;
    
    let d000 = distance_field[base_idx].distance;
    let d001 = distance_field[base_idx + 1u].distance;
    let d010 = distance_field[base_idx + gs].distance;
    let d011 = distance_field[base_idx + gs + 1u].distance;
    let d100 = distance_field[base_idx + gs_sq].distance;
    let d101 = distance_field[base_idx + gs_sq + 1u].distance;
    let d110 = distance_field[base_idx + gs_sq + gs].distance;
    let d111 = distance_field[base_idx + gs_sq + gs + 1u].distance;
    
    // Trilinear interpolation
    let fx = frac.x;
    let fy = frac.y; 
    let fz = frac.z;
    
    // Interpolate along X axis
    let d00 = mix(d000, d001, fx);
    let d01 = mix(d010, d011, fx);
    let d10 = mix(d100, d101, fx);
    let d11 = mix(d110, d111, fx);
    
    // Interpolate along Y axis
    let d0 = mix(d00, d01, fy);
    let d1 = mix(d10, d11, fy);
    
    // Interpolate along Z axis
    return mix(d0, d1, fz);
}

// Nearest-neighbor distance field sampling (fast version for when quality isn't critical)
fn sample_distance_field_nearest(pos: vec3<f32>) -> f32 {
    let gs_f = f32(context.grid_size);
    
    // Fast bounds check
    let in_bounds = all(pos >= vec3<f32>(0.0)) && all(pos < vec3<f32>(gs_f));
    if (!in_bounds) {
        return 1.0;
    }
    
    // Nearest neighbor sampling
    let grid_pos = clamp(pos, vec3<f32>(0.0), vec3<f32>(gs_f - 1.0));
    let nearest_pos = round(grid_pos);
    
    let gs = context.grid_size;
    let index = u32(nearest_pos.z) * gs * gs + u32(nearest_pos.y) * gs + u32(nearest_pos.x);
    
    return distance_field[index].distance;
}

// Adaptive distance field sampling based on distance from camera and surface proximity
fn sample_distance_field_adaptive(pos: vec3<f32>, camera_pos: vec3<f32>, surface_distance: f32) -> f32 {
    let camera_distance = distance(pos, camera_pos);
    
    // Use higher quality sampling when close to camera or near surfaces
    let use_high_quality = camera_distance < 20.0 || abs(surface_distance) < 2.0;
    
    return select(sample_distance_field_nearest(pos), sample_distance_field(pos), use_high_quality);
}

// Hierarchical distance field sampling - start coarse, refine near surface
fn sample_distance_field_hierarchical(pos: vec3<f32>, target_precision: f32) -> f32 {
    // Start with coarse sampling for distant areas
    if (target_precision > 1.0) {
        // Use nearest neighbor for distant sampling
        return sample_distance_field_nearest(pos);
    } else {
        // Use full trilinear interpolation for precise sampling
        return sample_distance_field(pos);
    }
}

// Fast empty space detection using lower resolution sampling
fn is_empty_region(pos: vec3<f32>, radius: f32) -> bool {
    // Sample at reduced resolution to quickly check if region is empty
    let coarse_dist = sample_distance_field_nearest(pos);
    return coarse_dist > radius;
}

// Calculate smooth gradient-based normal using central differences
fn calculate_gradient_normal(pos: vec3<f32>) -> vec3<f32> {
    let epsilon = 0.1; // Small offset for finite differences
    
    // Sample distance field at offset positions to calculate gradient
    let dx = sample_distance_field(pos + vec3<f32>(epsilon, 0.0, 0.0)) - 
             sample_distance_field(pos - vec3<f32>(epsilon, 0.0, 0.0));
    let dy = sample_distance_field(pos + vec3<f32>(0.0, epsilon, 0.0)) - 
             sample_distance_field(pos - vec3<f32>(0.0, epsilon, 0.0));
    let dz = sample_distance_field(pos + vec3<f32>(0.0, 0.0, epsilon)) - 
             sample_distance_field(pos - vec3<f32>(0.0, 0.0, epsilon));
    
    let gradient = vec3<f32>(dx, dy, dz) / (2.0 * epsilon);
    let gradient_length = length(gradient);
    
    // Return normalized gradient, fallback to up vector if gradient is too small
    return select(vec3<f32>(0.0, 1.0, 0.0), gradient / gradient_length, gradient_length > 0.001);
}

// Calculate hard voxel face normal (axis-aligned) - kept for compatibility
fn calculate_voxel_normal(pos: vec3<f32>) -> vec3<f32> {
    let gs_f = f32(context.grid_size);
    let grid_pos = clamp(pos, vec3<f32>(0.0), vec3<f32>(gs_f - 1.0));
    let voxel_center = round(grid_pos);
    
    // Get the offset from voxel center to determine which face we hit
    let offset = pos - voxel_center;
    let abs_offset = abs(offset);
    
    // Determine which axis has the maximum offset (which face we're closest to)
    let is_x = f32(abs_offset.x >= abs_offset.y && abs_offset.x >= abs_offset.z);
    let is_y = f32(abs_offset.y >= abs_offset.x && abs_offset.y >= abs_offset.z);
    let is_z = 1.0 - is_x - is_y;
    
    // Return the face normal in the direction of the offset
    return sign(offset) * vec3<f32>(is_x, is_y, is_z);
}

// Optimized raymarching with sphere tracing and binary search refinement
fn raycast_distance_field(ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> RayCast {
    var result: RayCast;
    result.pos = vec3<f32>(-1.0);
    result.data = 0u;
    result.steps = 0u;
    
    var t = 0.0;
    let epsilon = context.sdf_epsilon;
    let max_steps = context.sdf_max_steps;
    let over_relaxation = context.sdf_over_relaxation;
    
    var prev_dist = 1.0;
    var prev_t = 0.0;
    
    // Phase 1: Sphere tracing with large steps
    for (var i: u32 = 0u; i < max_steps; i++) {
        let pos = ray_origin + ray_dir * t;
        let dist = sample_distance_field(pos);
        
        result.steps = i;
        
        // Check for surface crossing
        if (i > 0u && prev_dist >= 0.0 && dist < 0.0) {
            // Phase 2: Binary search refinement for precise intersection
            var t_near = prev_t;
            var t_far = t;
            var refined_pos = pos;
            
            // Binary search for precise surface location (up to 6 iterations)
            for (var j = 0u; j < 6u; j++) {
                let t_mid = (t_near + t_far) * 0.5;
                let mid_pos = ray_origin + ray_dir * t_mid;
                let mid_dist = sample_distance_field(mid_pos);
                
                if (mid_dist >= 0.0) {
                    t_near = t_mid;
                } else {
                    t_far = t_mid;
                    refined_pos = mid_pos;
                }
                result.steps += 1u;
            }
            
            result.pos = refined_pos;
            result.normal = calculate_gradient_normal(refined_pos);
            
            let grid_pos = clamp(refined_pos, vec3<f32>(0.0), vec3<f32>(f32(context.grid_size) - 1.0));
            let grid_i = min(vec3<u32>(round(grid_pos)), vec3<u32>(context.grid_size - 1u));
            let gs = context.grid_size;
            let index = grid_i.z * gs * gs + grid_i.y * gs + grid_i.x;
            
            result.data = distance_field[index].material_id;
            return result;
        }
        
        // Sphere tracing: use distance as step size with safety factor
        let abs_dist = abs(dist);
        var step_size: f32;
        
        if (abs_dist > 2.0) {
            // Large steps in empty space
            step_size = abs_dist * over_relaxation;
        } else if (abs_dist > 0.5) {
            // Medium steps approaching surface
            step_size = abs_dist * 0.5;
        } else {
            // Small steps near surface
            step_size = 0.1;
        }
        
        prev_t = t;
        t += step_size;
        prev_dist = dist;
        
        // Early termination
        if (t > SDF_MAX_DISTANCE) {
            break;
        }
    }
    
    return result;
}

// Ultra-optimized raymarching with all optimization techniques
fn raycast_distance_field_optimized(ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> RayCast {
    var result: RayCast;
    result.pos = vec3<f32>(-1.0);
    result.data = 0u;
    result.steps = 0u;
    
    var t = 0.0;
    let max_steps = min(context.sdf_max_steps, 64u); // Reduced max steps
    
    var prev_dist = 1.0;
    var prev_t = 0.0;
    var step_size = 4.0; // Start with large steps
    
    // Phase 1: Coarse traversal with large steps
    for (var i: u32 = 0u; i < max_steps; i++) {
        let pos = ray_origin + ray_dir * t;
        
        // Early bounds check
        if (any(pos < vec3<f32>(0.0)) || any(pos >= vec3<f32>(f32(context.grid_size)))) {
            break;
        }
        
        // Use hierarchical sampling based on step size
        let dist = sample_distance_field_hierarchical(pos, step_size);
        result.steps = i;
        
        // Surface crossing detection
        if (i > 0u && prev_dist >= 0.0 && dist < 0.0) {
            // Phase 2: Binary search refinement
            var t_near = prev_t;
            var t_far = t;
            var refined_pos = pos;
            
            // Adaptive binary search iterations based on step size
            let search_iterations = select(4u, 6u, step_size > 2.0);
            
            for (var j = 0u; j < search_iterations; j++) {
                let t_mid = (t_near + t_far) * 0.5;
                let mid_pos = ray_origin + ray_dir * t_mid;
                let mid_dist = sample_distance_field(mid_pos); // Use full precision
                
                if (mid_dist >= 0.0) {
                    t_near = t_mid;
                } else {
                    t_far = t_mid;
                    refined_pos = mid_pos;
                }
                result.steps += 1u;
            }
            
            result.pos = refined_pos;
            result.normal = calculate_gradient_normal(refined_pos);
            
            let grid_pos = clamp(refined_pos, vec3<f32>(0.0), vec3<f32>(f32(context.grid_size) - 1.0));
            let grid_i = min(vec3<u32>(round(grid_pos)), vec3<u32>(context.grid_size - 1u));
            let gs = context.grid_size;
            let index = grid_i.z * gs * gs + grid_i.y * gs + grid_i.x;
            
            result.data = distance_field[index].material_id;
            return result;
        }
        
        // Optimized step sizing with safety constraints
        let abs_dist = abs(dist);
        
        // Large steps in very empty space
        if (abs_dist > 8.0) {
            step_size = min(abs_dist * 0.9, 16.0);
        }
        // Medium steps in moderately empty space  
        else if (abs_dist > 3.0) {
            step_size = abs_dist * 0.7;
        }
        // Small steps approaching surface
        else if (abs_dist > 1.0) {
            step_size = abs_dist * 0.5;
        }
        // Very small steps near surface
        else {
            step_size = max(abs_dist * 0.3, 0.05);
        }
        
        prev_t = t;
        t += step_size;
        prev_dist = dist;
        
        // Early termination for distant rays
        if (t > SDF_MAX_DISTANCE) {
            break;
        }
        
        // Skip ahead in confirmed empty regions
        if (abs_dist > 10.0 && is_empty_region(pos + ray_dir * abs_dist * 0.5, abs_dist * 0.3)) {
            t += abs_dist * 0.5;
            result.steps += 1u;
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

// Enhanced hybrid raycast with adaptive sampling and smooth normals
fn raycast_hybrid(ray_origin: vec3<f32>, ray_dir: vec3<f32>, grid_size: u32) -> RayCast {
    var result: RayCast;
    result.pos = vec3<f32>(-1.0);
    result.data = 0u;
    result.steps = 0u;

    let gs_f = f32(grid_size);
    let root_min = vec3<f32>(0.0);
    let root_max = vec3<f32>(gs_f);

    // Check if ray hits the grid at all
    let root_tt = ray_aabb_intersect(ray_origin, ray_dir, root_min, root_max);
    if (root_tt.x > root_tt.y) {
        return result;
    }

    var t = max(root_tt.x, 0.0);
    let max_distance = min(root_tt.y, SDF_MAX_DISTANCE);
    let max_steps = context.sdf_max_steps;
    
    // Start with large steps, reduce as we get closer to surfaces
    var step_size = 4.0; // Start with 4-voxel steps
    
    // Sample initial distance to determine if we start inside or outside
    let initial_pos = ray_origin + ray_dir * t;
    var prev_dist = sample_distance_field(initial_pos);
    
    for (var i: u32 = 0u; i < max_steps; i++) {
        let pos = ray_origin + ray_dir * t;
        
        // Early exit if outside bounds
        if (any(pos < root_min) || any(pos >= root_max)) {
            break;
        }
        
        // Use optimized sampling - avoid adaptive overhead in tight loops
        let dist = sample_distance_field(pos);
        result.steps = i;
        
        // Surface detection: handle both outside→inside and inside→outside transitions
        let surface_hit = (i > 0u && prev_dist >= 0.0 && dist < 0.0) ||  // Outside to inside (front face)
                         (i > 0u && prev_dist < 0.0 && dist >= 0.0);      // Inside to outside (back face)
        
        if (surface_hit) {
            // Binary search refinement for precise surface location
            var t_near = t - step_size;
            var t_far = t;
            var refined_pos = pos;
            
            // 4 iterations for hybrid (fewer than pure SDF since we start closer)
            for (var j = 0u; j < 4u; j++) {
                let t_mid = (t_near + t_far) * 0.5;
                let mid_pos = ray_origin + ray_dir * t_mid;
                let mid_dist = sample_distance_field(mid_pos);
                
                if (mid_dist >= 0.0) {
                    t_near = t_mid;
                } else {
                    t_far = t_mid;
                    refined_pos = mid_pos;
                }
                result.steps += 1u;
            }
            
            let grid_pos = clamp(refined_pos, vec3<f32>(0.0), vec3<f32>(gs_f - 1.0));
            result.pos = refined_pos;
            result.normal = calculate_gradient_normal(refined_pos);

            let grid_i = min(vec3<u32>(round(grid_pos)), vec3<u32>(grid_size - 1u));
            let gs = grid_size;
            let index = grid_i.z * gs * gs + grid_i.y * gs + grid_i.x;
            
            result.data = distance_field[index].material_id;
            return result;
        }
        
        // Optimized step sizing using distance field values
        let abs_dist = abs(dist);
        if (abs_dist > 3.0) {
            step_size = min(abs_dist * 0.8, 8.0); // Large steps, capped for safety
        } else if (abs_dist > 1.5) {
            step_size = abs_dist * 0.6; // Medium steps
        } else if (abs_dist > 0.5) {
            step_size = abs_dist * 0.4; // Smaller steps approaching surface
        } else {
            step_size = 0.1; // Very small steps near surface
        }
        
        t += step_size;
        prev_dist = dist;
        
        if (t > max_distance) {
            break;
        }
    }
    
    return result;
}

// SDF raymarching constrained to a specific region (AABB)
fn raycast_sdf_in_region(ray_origin: vec3<f32>, ray_dir: vec3<f32>, region_min: vec3<f32>, region_max: vec3<f32>) -> RayCast {
    var result: RayCast;
    result.pos = vec3<f32>(-1.0);
    result.data = 0u;
    result.steps = 0u;

    // Find intersection with the region
    let region_tt = ray_aabb_intersect(ray_origin, ray_dir, region_min, region_max);
    if (region_tt.x > region_tt.y) {
        return result;
    }

    var t = max(region_tt.x, 0.0);
    let t_max = region_tt.y;
    let epsilon = context.sdf_epsilon;
    let max_steps = min(context.sdf_max_steps / 8u, 32u); // Much fewer steps since we're in a small region
    let over_relaxation = context.sdf_over_relaxation;
    
    var prev_dist = 1.0;
    
    for (var i: u32 = 0u; i < max_steps; i++) {
        let pos = ray_origin + ray_dir * t;
        
        // Early exit if we've left the region
        if (any(pos < region_min) || any(pos >= region_max)) {
            break;
        }
        
        let dist = sample_distance_field(pos);
        result.steps = i;
        
        // Surface detection: transition from empty to solid
        if (i > 0u && prev_dist >= 0.0 && dist < 0.0) {
            result.pos = pos;
            result.normal = calculate_gradient_normal(pos);
            
            let grid_pos = clamp(pos, vec3<f32>(0.0), vec3<f32>(f32(context.grid_size) - 1.0));
            let grid_i = min(vec3<u32>(round(grid_pos)), vec3<u32>(context.grid_size - 1u));
            let gs = context.grid_size;
            let index = grid_i.z * gs * gs + grid_i.y * gs + grid_i.x;
            
            result.data = distance_field[index].material_id;
            return result;
        }
        
        // Adaptive stepping with region bounds
        let step_size = max(abs(dist) * over_relaxation, 0.005);
        t += step_size;
        prev_dist = dist;
        
        // Exit if we've reached the end of the region
        if (t > t_max) {
            break;
        }
    }
    
    return result;
}

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
        
        // Check if this is a leaf or minimum size
        if (entry.size <= 1.0) {
            let data = node.data;
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

        // Compute child mask from childs array
        var child_mask = 0u;
        for (var i = 0u; i < 8u; i++) {
            if (node.childs[i] != INVALID_INDEX) {
                child_mask |= (1u << i);
            }
        }
        
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

            let child_node_index = node.childs[octant];
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
  @location(2) worldPosition: vec4<f32>
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

  // Raycast through octree, distance field, or hybrid based on render mode
  var hit: RayCast;
  if (context.render_mode >= 4u) {
    // Use hybrid raymarching (modes 4 and 5)
    hit = raycast_hybrid(camera_pos, ray_dir, context.grid_size);
  } else if (context.render_mode >= 2u) {
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
  if (context.render_mode == 0u || context.render_mode == 2u || context.render_mode == 4u) {
    // Normal rendering mode (octree, SDF, or hybrid)
    output.colorForTexture = final_color;
    output.colorForCanvas = final_color;
  } else {
    // Heatmap rendering mode (octree, SDF, or hybrid)
    let heatmap_output = vec4(heatmap_color, 1.0);
    output.colorForTexture = heatmap_output;
    output.colorForCanvas = heatmap_output;
  }
  
  // Store world position for next frame's TAA (w component stores hit validity)
  output.worldPosition = vec4<f32>(hit.pos, select(-1.0, 1.0, hit.pos.x >= -0.001));
  return output;
}
