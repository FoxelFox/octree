#import "../data/context.wgsl"

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

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = frag_coord.xy / context.resolution;
    let pixel_coord = vec2<i32>(frag_coord.xy);
    
    // Sample G-buffer data
    let position_data = textureLoad(positionTexture, pixel_coord, 0);
    let normal_data = textureLoad(normalTexture, pixel_coord, 0);
    let diffuse_data = textureLoad(diffuseTexture, pixel_coord, 0);
    let depth = textureLoad(depthTexture, pixel_coord, 0);
    
    // Check if this pixel has geometry (depth test)
    let has_geometry = depth < 1.0;
    
    if (!has_geometry) {
        // Background pixel - use the beautiful colored background from post.wgsl
        let ndc = vec4<f32>((uv - 0.5) * 2.0 * vec2<f32>(1.0, -1.0), 0.0, 1.0);
        let view_pos = context.inverse_perspective * ndc;
        let world_pos = context.inverse_view * vec4<f32>(view_pos.xyz / view_pos.w, 1.0);
        let camera_pos = context.inverse_view[3].xyz;
        let ray_dir = normalize(world_pos.xyz - camera_pos);
        
        return vec4<f32>(abs(ray_dir) * 0.5 + 0.5, 1.0);
    }
    
    // Extract G-buffer data
    let world_pos = position_data.xyz;
    let distance = position_data.w;
    let world_normal = normalize(normal_data.xyz);
    let diffuse_color = diffuse_data.xyz;
    
    // Extract camera position from inverse view matrix
    let camera_pos = context.inverse_view[3].xyz;
    
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
    
    return vec4<f32>(final_color, 1.0);
}