#import "../data/context.wgsl"

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

// Calculate motion vectors for TAA (adapted from post.wgsl)
fn calculate_motion_vector(world_pos: vec3<f32>, current_uv: vec2<f32>, ray_dir: vec3<f32>, camera_pos: vec3<f32>) -> vec2<f32> {
    let has_hit = world_pos.x >= -0.001;
    let far_distance = 10000.0;
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
    let is_small_motion = motion_magnitude < 0.001;
    
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
    let has_current_hit = world_hit_pos.x >= -0.001;
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
                    if (depth_diff > 0.1) { // Significant depth change
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
        let rejection_threshold = max(0.1, depth * 0.002);
        if (world_pos_diff > rejection_threshold) {
            is_history_valid = false;
        }
        
        // Strict surface normal validation
        let current_normal = normalize(textureLoad(normalTexture, pixel_coord, 0).xyz);
        let prev_coord = vec2<i32>(history_uv * context.resolution);
        if (all(prev_coord >= vec2(0)) && all(prev_coord < vec2<i32>(context.resolution))) {
            let prev_normal = normalize(textureLoad(normalTexture, prev_coord, 0).xyz);
            let normal_similarity = dot(current_normal, prev_normal);
            if (normal_similarity < 0.9) { // Very strict normal similarity
                is_history_valid = false;
            }
        }
    }
    
    // Motion-based rejection - reject history for any significant motion
    let motion_magnitude = length(motion_vector);
    if (motion_magnitude > 0.01) { // Reject for motion > 1% of screen
        is_history_valid = false;
    }
    
    // Simple neighborhood clamping in RGB space (lighter weight)
    var color_min = current_color.rgb;
    var color_max = current_color.rgb;
    
    // Sample 3x3 neighborhood for clamping bounds
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let sample_coord = pixel_coord + vec2(dx, dy);
            let bounds_check = all(sample_coord >= vec2(0)) && all(sample_coord < vec2<i32>(context.resolution));
            
            if (bounds_check) {
                let sample_depth = textureLoad(depthTexture, sample_coord, 0);
                let sample_has_geo = sample_depth < 1.0;
                
                // Only use neighbors with similar geometry state for clamping
                if (sample_has_geo == center_has_geo) {
                    var neighbor_color: vec3<f32>;
                    
                    if (sample_has_geo) {
                        // Sample G-buffer and reconstruct lighting
                        let pos_data = textureLoad(positionTexture, sample_coord, 0);
                        let normal_data = textureLoad(normalTexture, sample_coord, 0);
                        let diffuse_data = textureLoad(diffuseTexture, sample_coord, 0);
                        
                        let world_pos = pos_data.xyz;
                        let distance = pos_data.w;
                        let world_normal = normalize(normal_data.xyz);
                        let diffuse_color = diffuse_data.xyz;
                        
                        // Apply lighting
                        let max_light_distance = 128.0;
                        var light_intensity = 0.7;
                        if (distance > 0.0) {
                            let falloff_factor = max(0.0, (max_light_distance - distance) / max_light_distance);
                            light_intensity = falloff_factor * falloff_factor;
                        }
                        
                        let light_dir = normalize(camera_pos - world_pos);
                        let diffuse = max(dot(world_normal, light_dir), 0.0);
                        let ambient = 0.3;
                        let lighting = ambient + diffuse * light_intensity;
                        let lit_color = diffuse_color * lighting;
                        
                        // Apply fog
                        let view_ray = normalize(world_pos - camera_pos);
                        let fog_color = abs(view_ray) * 0.5 + 0.5;
                        let fog_factor = clamp((256.0 - distance) / 206.0, 0.0, 1.0);
                        neighbor_color = mix(fog_color, lit_color, fog_factor);
                    } else {
                        // Background color
                        let ndc = vec4<f32>((vec2<f32>(sample_coord) / context.resolution - 0.5) * 2.0 * vec2<f32>(1.0, -1.0), 0.0, 1.0);
                        let view_pos = context.inverse_perspective * ndc;
                        let world_pos_bg = context.inverse_view * vec4<f32>(view_pos.xyz / view_pos.w, 1.0);
                        let ray_dir_sample = normalize(world_pos_bg.xyz - camera_pos);
                        neighbor_color = abs(ray_dir_sample) * 0.5 + 0.5;
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
    
    // Sample G-buffer data
    let position_data = textureLoad(positionTexture, pixel_coord, 0);
    let normal_data = textureLoad(normalTexture, pixel_coord, 0);
    let diffuse_data = textureLoad(diffuseTexture, pixel_coord, 0);
    let depth = textureLoad(depthTexture, pixel_coord, 0);
    
    // Check if this pixel has geometry (depth test)
    let has_geometry = depth < 1.0;
    
    // Calculate background data (needed for both paths)
    let ndc = vec4<f32>((uv - 0.5) * 2.0 * vec2<f32>(1.0, -1.0), 0.0, 1.0);
    let view_pos = context.inverse_perspective * ndc;
    let world_pos_bg = context.inverse_view * vec4<f32>(view_pos.xyz / view_pos.w, 1.0);
    let camera_pos = context.inverse_view[3].xyz;
    let ray_dir = normalize(world_pos_bg.xyz - camera_pos);
    let background_color = vec4<f32>(abs(ray_dir) * 0.5 + 0.5, 1.0);
    
    // Pre-sample TAA history for both background and geometry (in uniform control flow)
    let motion_vector_bg = calculate_motion_vector(vec3<f32>(-1.0), uv, ray_dir, camera_pos);
    let history_uv_bg = uv + motion_vector_bg;
    
    // Extract geometry data
    let world_pos_geo = position_data.xyz;
    let motion_vector_geo = calculate_motion_vector(world_pos_geo, uv, normalize(world_pos_geo - camera_pos), camera_pos);
    let history_uv_geo = uv + motion_vector_geo;
    
    // Sample history textures for both paths (uniform control flow)
    let history_color_bg = textureSample(prevFrameTexture, texSampler, history_uv_bg);
    let history_color_geo = textureSample(prevFrameTexture, texSampler, history_uv_geo);
    
    if (!has_geometry) {
        // Background pixel processing
        let taa_background = taa_sample_history_presampled(uv, background_color, vec3<f32>(-1.0), camera_pos, ray_dir, history_color_bg, motion_vector_bg);
        
        var output: FragmentOutput;
        output.colorForCanvas = taa_background;
        output.colorForTexture = taa_background;
        output.worldPosition = vec4<f32>(vec3<f32>(-1.0), -1.0); // Invalid world position for background
        return output;
    }
    
    // Extract G-buffer data
    let world_pos = position_data.xyz;
    let distance = position_data.w;
    let world_normal = normalize(normal_data.xyz);
    let diffuse_color = diffuse_data.xyz;
    
    // Light falloff parameters
    let max_light_distance = 128.0; // Maximum distance for light effect
    let falloff_start = 0.0; // Distance where falloff begins
    
    // Calculate distance attenuation
    var light_intensity = 0.7;
    if (distance > falloff_start) {
        let falloff_range = max_light_distance - falloff_start;
        let falloff_factor = max(0.0, (max_light_distance - distance) / falloff_range);
        light_intensity = falloff_factor * falloff_factor; // Quadratic falloff
    }
    
    // Light direction from fragment to camera (light at camera position)
    let light_dir = normalize(camera_pos - world_pos);
    
    // Simple diffuse lighting
    let diffuse = max(dot(world_normal, light_dir), 0.0);
    
    // Ambient lighting to prevent completely black surfaces
    let ambient = 0.3;
    
    // Apply distance attenuation to diffuse lighting only
    let lighting = ambient + diffuse * light_intensity;
    
    // Apply lighting to color
    let lit_color = diffuse_color * lighting;
    
    // Fog parameters - using beautiful background color from post.wgsl
    let view_ray = normalize(world_pos - camera_pos);
    let fog_color = abs(view_ray) * 0.5 + 0.5; // Same beautiful colors as post.wgsl background
    let fog_start = 50.0;
    let fog_end = 256.0;
    
    // Calculate fog factor (0 = full fog, 1 = no fog)
    let fog_factor = clamp((fog_end - distance) / (fog_end - fog_start), 0.0, 1.0);
    
    // Mix lit color with fog color
    let final_color = mix(fog_color, lit_color, fog_factor);
    let final_result = vec4<f32>(final_color, 1.0);
    
    // Apply TAA if enabled
    let ray_dir_geo = normalize(world_pos - camera_pos);
    let taa_result = select(final_result, taa_sample_history_presampled(uv, final_result, world_pos, camera_pos, ray_dir_geo, history_color_geo, motion_vector_geo), context.taa_enabled != 0u);
    
    var output: FragmentOutput;
    output.colorForCanvas = taa_result;
    output.colorForTexture = taa_result;
    output.worldPosition = vec4<f32>(world_pos, select(-1.0, 1.0, world_pos.x >= -0.001)); // Store world position for next frame's TAA
    
    return output;
}