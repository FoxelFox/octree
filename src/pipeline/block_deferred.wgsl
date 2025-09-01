#import "../data/context.wgsl"

// Constants
const HIT_THRESHOLD = -0.001;
const FAR_DISTANCE = 10000.0;
const MAX_LIGHT_DISTANCE = 128.0;
const AMBIENT_LIGHT = 0.3;
const BASE_LIGHT_INTENSITY = 0.7;
const FOG_START = 50.0;
const FOG_END = 256.0;
const MOTION_THRESHOLD = 0.001;
const MOTION_REJECT_THRESHOLD = 0.01;
const DEPTH_THRESHOLD = 0.1;
const NORMAL_SIMILARITY_THRESHOLD = 0.9;
const WORLD_POS_REJECTION_BASE = 0.1;
const WORLD_POS_REJECTION_SCALE = 0.002;

@group(0) @binding(0) var<uniform> context: Context;
@group(1) @binding(0) var positionTexture: texture_2d<f32>;
@group(1) @binding(1) var normalTexture: texture_2d<f32>;
@group(1) @binding(2) var diffuseTexture: texture_2d<f32>;
@group(1) @binding(3) var depthTexture: texture_depth_2d;
@group(2) @binding(0) var prevFrameTexture: texture_2d<f32>;
@group(2) @binding(1) var texSampler: sampler;
@group(2) @binding(2) var prevWorldPosTexture: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    // Full-screen quad
    const pos = array(
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
        vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
    );
    return vec4(pos[i], 0.0, 1.0);
}

// Calculate background color and ray direction from UV coordinates
fn calculate_background_color_and_ray(uv: vec2<f32>, camera_pos: vec3<f32>) -> vec4<f32> {
    let ndc = vec4<f32>((uv - 0.5) * 2.0 * vec2<f32>(1.0, -1.0), 0.0, 1.0);
    let view_pos = context.inverse_perspective * ndc;
    let world_pos_bg = context.inverse_view * vec4<f32>(view_pos.xyz / view_pos.w, 1.0);
    let ray_dir = normalize(world_pos_bg.xyz - camera_pos);
    return vec4<f32>(abs(ray_dir) * 0.5 + 0.5, 1.0);
}

// Calculate background color from ray direction
fn calculate_background_color(uv: vec2<f32>, camera_pos: vec3<f32>) -> vec4<f32> {
    return calculate_background_color_and_ray(uv, camera_pos);
}

// Calculate ray direction from UV coordinates
fn calculate_ray_direction(uv: vec2<f32>, camera_pos: vec3<f32>) -> vec3<f32> {
    let ndc = vec4<f32>((uv - 0.5) * 2.0 * vec2<f32>(1.0, -1.0), 0.0, 1.0);
    let view_pos = context.inverse_perspective * ndc;
    let world_pos_bg = context.inverse_view * vec4<f32>(view_pos.xyz / view_pos.w, 1.0);
    return normalize(world_pos_bg.xyz - camera_pos);
}

// Calculate lighting for a given world position and surface properties
fn calculate_lighting(world_pos: vec3<f32>, world_normal: vec3<f32>, diffuse_color: vec3<f32>, camera_pos: vec3<f32>, distance: f32) -> vec3<f32> {
    // Calculate distance attenuation
    var light_intensity = BASE_LIGHT_INTENSITY;
    if (distance > 0.0) {
        let falloff_factor = max(0.0, (MAX_LIGHT_DISTANCE - distance) / MAX_LIGHT_DISTANCE);
        light_intensity = falloff_factor * falloff_factor;
    }
    
    // Light direction from fragment to camera (light at camera position)
    let light_dir = normalize(camera_pos - world_pos);
    let diffuse = max(dot(world_normal, light_dir), 0.0);
    let lighting = AMBIENT_LIGHT + diffuse * light_intensity;
    
    // Apply lighting to color
    let lit_color = diffuse_color * lighting;
    
    // Apply fog
    let view_ray = normalize(world_pos - camera_pos);
    let fog_color = abs(view_ray) * 0.5 + 0.5;
    let fog_factor = clamp((FOG_END - distance) / (FOG_END - FOG_START), 0.0, 1.0);
    
    return mix(fog_color, lit_color, fog_factor);
}

// Calculate motion vectors for TAA (adapted from post.wgsl)
fn calculate_motion_vector(world_pos: vec3<f32>, current_uv: vec2<f32>, ray_dir: vec3<f32>, camera_pos: vec3<f32>) -> vec2<f32> {
    let has_hit = world_pos.x >= HIT_THRESHOLD;
    let far_distance = FAR_DISTANCE;
    let pos_for_motion = select(ray_dir * far_distance, world_pos, has_hit);

    // Create non-jittered projection by removing jitter from the current perspective matrix
    var unjittered_perspective = context.perspective;
    unjittered_perspective[2][0] -= context.jitter_offset.x; // Remove jitter from x offset
    unjittered_perspective[2][1] -= context.jitter_offset.y; // Remove jitter from y offset
    
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
    let is_small_motion = motion_magnitude < MOTION_THRESHOLD;
    
    return select(motion_vec, vec2<f32>(0.0), !has_hit && is_small_motion);
}

// Convert RGB to YCoCg color space for better variance analysis
fn rgb_to_ycocg(color: vec3<f32>) -> vec3<f32> {
    let y = dot(color, vec3<f32>(0.25, 0.5, 0.25));
    let co = dot(color, vec3<f32>(0.5, 0.0, -0.5));
    let cg = dot(color, vec3<f32>(-0.25, 0.5, -0.25));
    return vec3<f32>(y, co, cg);
}

// Convert YCoCg back to RGB
fn ycocg_to_rgb(ycocg: vec3<f32>) -> vec3<f32> {
    let tmp = ycocg.x - ycocg.z;
    let g = ycocg.z + tmp;
    let b = tmp - ycocg.y;
    let r = b + ycocg.y;
    return vec3<f32>(r, g, b);
}

// Conservative TAA focused on anti-aliasing with minimal smearing
fn taa_sample_history_presampled(current_uv: vec2<f32>, current_color: vec4<f32>, world_hit_pos: vec3<f32>, camera_pos: vec3<f32>, ray_dir: vec3<f32>, history_color: vec4<f32>, motion_vector: vec2<f32>) -> vec4<f32> {
    let has_current_hit = world_hit_pos.x >= HIT_THRESHOLD;
    let history_uv = current_uv + motion_vector;
    let pixel_coord = vec2<i32>(current_uv * context.resolution);

    // Sample previous frame data for validation
    let history_coord_i = vec2<i32>(clamp(history_uv, vec2(0.0), vec2(0.999)) * context.resolution);
    let prev_frame_data = textureLoad(prevWorldPosTexture, history_coord_i, 0);
    
    // Detect actual aliasing edges where TAA is beneficial
    var is_aliased_edge = false;
    let center_depth = textureLoad(depthTexture, pixel_coord, 0);
    let center_has_geo = center_depth < 1.0;
    
    // Check for geometry/background transitions (main source of aliasing)
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) { continue; }
            let sample_coord = pixel_coord + vec2(dx, dy);
            let bounds_check = all(sample_coord >= vec2(0)) && all(sample_coord < vec2<i32>(context.resolution));
            
            if (bounds_check) {
                let neighbor_depth = textureLoad(depthTexture, sample_coord, 0);
                let neighbor_has_geo = neighbor_depth < 1.0;
                
                // Geometry/background edge - prime candidate for aliasing
                if (center_has_geo != neighbor_has_geo) {
                    is_aliased_edge = true;
                    break;
                }
                
                // Sharp depth discontinuities within geometry
                if (center_has_geo && neighbor_has_geo) {
                    let depth_diff = abs(center_depth - neighbor_depth);
                    if (depth_diff > DEPTH_THRESHOLD) {
                        is_aliased_edge = true;
                        break;
                    }
                }
            }
        }
        if (is_aliased_edge) { break; }
    }
    
    // Skip TAA entirely for non-aliased areas (most pixels)
    if (!is_aliased_edge) {
        return current_color;
    }
    
    // Very conservative history validation - reject easily to prevent smearing
    let is_in_bounds = all(history_uv >= vec2<f32>(0.0)) && all(history_uv <= vec2<f32>(1.0));
    let prev_data_valid = abs(prev_frame_data.w) > 0.5;
    let had_prev_hit = prev_frame_data.w > 0.0;
    var is_history_valid = is_in_bounds && prev_data_valid && (has_current_hit == had_prev_hit);

    // Extremely strict validation for geometry pixels
    if (is_history_valid && has_current_hit) {
        let prev_world_pos = prev_frame_data.xyz;
        let world_pos_diff = distance(world_hit_pos, prev_world_pos);
        let depth = distance(world_hit_pos, camera_pos);
        
        // Very tight rejection threshold - prefer sharp over smeared
        let rejection_threshold = max(WORLD_POS_REJECTION_BASE, depth * WORLD_POS_REJECTION_SCALE);
        if (world_pos_diff > rejection_threshold) {
            is_history_valid = false;
        }
        
        // Strict surface normal validation
        let current_normal = normalize(textureLoad(normalTexture, pixel_coord, 0).xyz);
        let prev_coord = vec2<i32>(history_uv * context.resolution);
        if (all(prev_coord >= vec2(0)) && all(prev_coord < vec2<i32>(context.resolution))) {
            let prev_normal = normalize(textureLoad(normalTexture, prev_coord, 0).xyz);
            let normal_similarity = dot(current_normal, prev_normal);
            if (normal_similarity < NORMAL_SIMILARITY_THRESHOLD) {
                is_history_valid = false;
            }
        }
    }
    
    // Motion-based rejection - reject history for any significant motion
    let motion_magnitude = length(motion_vector);
    if (motion_magnitude > MOTION_REJECT_THRESHOLD) {
        is_history_valid = false;
    }
    
    // Optimized neighborhood clamping - use pre-computed frame texture instead of G-buffer reconstruction
    var color_min = current_color.rgb;
    var color_max = current_color.rgb;
    
    // Sample 3x3 neighborhood for clamping bounds using the current frame's computed colors
    // This avoids expensive G-buffer reconstruction and lighting calculations
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) { continue; }
            
            let sample_coord = pixel_coord + vec2(dx, dy);
            let bounds_check = all(sample_coord >= vec2(0)) && all(sample_coord < vec2<i32>(context.resolution));
            
            if (bounds_check) {
                let sample_depth = textureLoad(depthTexture, sample_coord, 0);
                let sample_has_geo = sample_depth < 1.0;
                
                // Only use neighbors with similar geometry state for clamping
                if (sample_has_geo == center_has_geo) {
                    var neighbor_color: vec3<f32>;
                    
                    if (sample_has_geo) {
                        // Use optimized lighting function instead of inlining
                        let pos_data = textureLoad(positionTexture, sample_coord, 0);
                        let normal_data = textureLoad(normalTexture, sample_coord, 0);
                        let diffuse_data = textureLoad(diffuseTexture, sample_coord, 0);
                        
                        neighbor_color = calculate_lighting(pos_data.xyz, normalize(normal_data.xyz), diffuse_data.xyz, camera_pos, pos_data.w);
                    } else {
                        // Use optimized background function
                        let sample_uv = vec2<f32>(sample_coord) / context.resolution;
                        neighbor_color = calculate_background_color(sample_uv, camera_pos).rgb;
                    }
                    
                    color_min = min(color_min, neighbor_color);
                    color_max = max(color_max, neighbor_color);
                }
            }
        }
    }
    
    // Tight clamping to prevent color bleeding
    let clamped_history = vec4<f32>(clamp(history_color.rgb, color_min, color_max), history_color.a);
    
    // More aggressive history usage for better temporal stability
    let blend_factor = select(0.95, 0.99, is_history_valid); // 99% history when valid, 50% when invalid

    return mix(current_color, clamped_history, blend_factor);
}

// Output structure for TAA
struct FragmentOutput {
  @location(0) colorForCanvas: vec4<f32>,
  @location(1) colorForTexture: vec4<f32>,
  @location(2) worldPosition: vec4<f32>
};

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> FragmentOutput {
    let uv = frag_coord.xy / context.resolution;
    let pixel_coord = vec2<i32>(frag_coord.xy);
    
    // Sample G-buffer data once
    let position_data = textureLoad(positionTexture, pixel_coord, 0);
    let depth = textureLoad(depthTexture, pixel_coord, 0);
    let has_geometry = depth < 1.0;
    
    let camera_pos = context.inverse_view[3].xyz;
    
    // Calculate data for both background and geometry in uniform control flow
    let background_color = calculate_background_color(uv, camera_pos);
    let ray_dir_bg = calculate_ray_direction(uv, camera_pos);
    let motion_vector_bg = calculate_motion_vector(vec3<f32>(-1.0), uv, ray_dir_bg, camera_pos);
    
    // Sample G-buffer data for geometry path
    let normal_data = textureLoad(normalTexture, pixel_coord, 0);
    let diffuse_data = textureLoad(diffuseTexture, pixel_coord, 0);
    let world_pos = position_data.xyz;
    let distance = position_data.w;
    let world_normal = normalize(normal_data.xyz);
    let diffuse_color = diffuse_data.xyz;
    
    // Calculate geometry data in uniform control flow
    let final_color_geo = calculate_lighting(world_pos, world_normal, diffuse_color, camera_pos, distance);
    let final_result_geo = vec4<f32>(final_color_geo, 1.0);
    let ray_dir_geo = normalize(world_pos - camera_pos);
    let motion_vector_geo = calculate_motion_vector(world_pos, uv, ray_dir_geo, camera_pos);
    
    // Sample history textures in uniform control flow for both paths
    let history_color_bg = textureSample(prevFrameTexture, texSampler, uv + motion_vector_bg);
    let history_color_geo = textureSample(prevFrameTexture, texSampler, uv + motion_vector_geo);
    
    // Apply TAA for both paths in uniform control flow
    let taa_result_bg = select(background_color, taa_sample_history_presampled(uv, background_color, vec3<f32>(-1.0), camera_pos, ray_dir_bg, history_color_bg, motion_vector_bg), context.taa_enabled != 0u);
    let taa_result_geo = select(final_result_geo, taa_sample_history_presampled(uv, final_result_geo, world_pos, camera_pos, ray_dir_geo, history_color_geo, motion_vector_geo), context.taa_enabled != 0u);
    
    // Select final result based on geometry presence
    let final_taa_result = select(taa_result_geo, taa_result_bg, !has_geometry);
    let final_world_pos = select(vec4<f32>(world_pos, select(-1.0, 1.0, world_pos.x >= HIT_THRESHOLD)), vec4<f32>(vec3<f32>(-1.0), -1.0), !has_geometry);
    
    var output: FragmentOutput;
    output.colorForCanvas = final_taa_result;
    output.colorForTexture = final_taa_result;
    output.worldPosition = final_world_pos;
    
    return output;
}