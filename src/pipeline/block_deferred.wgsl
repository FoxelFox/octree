#import "../data/context.wgsl"

// Constants
const HIT_THRESHOLD = -0.001;
const FAR_DISTANCE = 10000.0;
const MAX_LIGHT_DISTANCE = 128.0;
const AMBIENT_LIGHT = 0.3;
const BASE_LIGHT_INTENSITY = 0.7;
const SUNLIGHT_INTENSITY = 0.8;
const SUNLIGHT_DIRECTION = vec3<f32>(-0.3, -0.7, 0.2);
const FOG_START = 50.0;
const FOG_END = 256.0;

@group(0) @binding(0) var<uniform> context: Context;
@group(1) @binding(0) var positionTexture: texture_2d<f32>;
@group(1) @binding(1) var normalTexture: texture_2d<f32>;
@group(1) @binding(2) var diffuseTexture: texture_2d<f32>;
@group(1) @binding(3) var depthTexture: texture_depth_2d;

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
    // Calculate distance attenuation for camera light
    var camera_light_intensity = BASE_LIGHT_INTENSITY;
    if (distance > 0.0) {
        let falloff_factor = max(0.0, (MAX_LIGHT_DISTANCE - distance) / MAX_LIGHT_DISTANCE);
        camera_light_intensity = falloff_factor * falloff_factor;
    }
    
    // Camera light direction from fragment to camera (light at camera position)
    let camera_light_dir = normalize(camera_pos - world_pos);
    let camera_diffuse = max(dot(world_normal, camera_light_dir), 0.0);
    
    // Sunlight direction (directional light)
    let sun_light_dir = normalize(SUNLIGHT_DIRECTION);
    let sun_diffuse = max(dot(world_normal, sun_light_dir), 0.0);
    
    // Combine lighting
    let lighting = AMBIENT_LIGHT + (camera_diffuse * camera_light_intensity) + (sun_diffuse * SUNLIGHT_INTENSITY);
    
    // Apply lighting to color
    let lit_color = diffuse_color * lighting;
    
    // Apply fog
    let view_ray = normalize(world_pos - camera_pos);
    let fog_color = abs(view_ray) * 0.5 + 0.5;
    let fog_factor = clamp((FOG_END - distance) / (FOG_END - FOG_START), 0.0, 1.0);
    
    return mix(fog_color, lit_color, fog_factor);
}


@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = frag_coord.xy / context.resolution;
    let pixel_coord = vec2<i32>(frag_coord.xy);
    
    // Sample G-buffer data
    let position_data = textureLoad(positionTexture, pixel_coord, 0);
    let depth = textureLoad(depthTexture, pixel_coord, 0);
    let has_geometry = depth < 1.0;
    
    let camera_pos = context.inverse_view[3].xyz;
    
    if (has_geometry) {
        // Render geometry with lighting
        let normal_data = textureLoad(normalTexture, pixel_coord, 0);
        let diffuse_data = textureLoad(diffuseTexture, pixel_coord, 0);
        let world_pos = position_data.xyz;
        let distance = position_data.w;
        let world_normal = normalize(normal_data.xyz);
        let diffuse_color = diffuse_data.xyz;
        
        let lit_color = calculate_lighting(world_pos, world_normal, diffuse_color, camera_pos, distance);
        return vec4<f32>(lit_color, 1.0);
    } else {
        // Render background
        return calculate_background_color(uv, camera_pos);
    }
}