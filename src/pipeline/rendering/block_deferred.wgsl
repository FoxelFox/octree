#import "../../data/context.wgsl"

// Constants
const FOG_START = 128.0;
const FOG_END = 512.0;

@group(0) @binding(0) var<uniform> context: Context;
@group(0) @binding(1) var<uniform> chunk_world_pos: vec4<i32>;
@group(1) @binding(0) var positionTexture: texture_2d<f32>;
@group(1) @binding(1) var normalTexture: texture_2d<f32>;
@group(1) @binding(2) var diffuseTexture: texture_2d<f32>;
@group(1) @binding(3) var depthTexture: texture_depth_2d;
@group(2) @binding(0) var space_background_texture: texture_2d<f32>;

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

// Convert ray direction to UV for texture sampling
fn ray_direction_to_uv(ray_dir: vec3<f32>) -> vec2<f32> {
    // Convert cartesian to spherical coordinates
    let theta = atan2(ray_dir.z, ray_dir.x); // Azimuth
    let phi = acos(ray_dir.y); // Elevation

    // Convert to UV coordinates
    let u = (theta + 3.14159265) / (2.0 * 3.14159265); // 0 to 1
    let v = phi / 3.14159265; // 0 to 1

    return vec2<f32>(u, v);
}

// Sample space background with separate nebula and stars
fn sample_space_background(ray_dir: vec3<f32>) -> vec3<f32> {
    let uv = ray_direction_to_uv(ray_dir);
    let texture_size = textureDimensions(space_background_texture);
    let coord = vec2<i32>(uv * vec2<f32>(texture_size));
    let data = textureLoad(space_background_texture, coord, 0);

    let nebula = data.rgb;
    let star_intensity = data.a;

    // Reconstruct star color from intensity (approximate original colors)
    let star_color = vec3<f32>(1.0, 0.9, 0.8) * star_intensity;

    return nebula + star_color;
}

// Sample only nebula for fog (no stars)
fn sample_nebula_only(ray_dir: vec3<f32>) -> vec3<f32> {
    let uv = ray_direction_to_uv(ray_dir);
    let texture_size = textureDimensions(space_background_texture);
    let coord = vec2<i32>(uv * vec2<f32>(texture_size));
    return textureLoad(space_background_texture, coord, 0).rgb; // Only nebula
}

// Calculate background color from ray direction
fn calculate_background_color(uv: vec2<f32>, camera_pos: vec3<f32>) -> vec4<f32> {
    let ray_dir = calculate_ray_direction(uv, camera_pos);
    let space_color = sample_space_background(ray_dir);
    return vec4<f32>(space_color, 1.0);
}

// Calculate ray direction from UV coordinates
fn calculate_ray_direction(uv: vec2<f32>, camera_pos: vec3<f32>) -> vec3<f32> {
    let ndc = vec4<f32>((uv - 0.5) * 2.0 * vec2<f32>(1.0, -1.0), 0.0, 1.0);
    let view_pos = context.inverse_perspective * ndc;
    let world_pos_bg = context.inverse_view * vec4<f32>(view_pos.xyz / view_pos.w, 1.0);
    return normalize(world_pos_bg.xyz - camera_pos);
}

// Apply fog and environment to baked vertex colors
fn apply_fog_and_environment(diffuse_color: vec3<f32>, light_visibility: f32, world_normal: vec3<f32>, camera_pos: vec3<f32>, world_pos: vec3<f32>, distance: f32) -> vec3<f32> {
    // Add environment reflection scaled by light visibility (1 - shadow)
    let env_reflection = sample_nebula_only(world_normal) * 0.08 * light_visibility;
    let final_color = diffuse_color + diffuse_color * env_reflection;

    // Apply space fog
    let view_ray = normalize(world_pos - camera_pos);
    let fog_color = sample_nebula_only(view_ray) * 0.8;
    let fog_factor = clamp((FOG_END - distance) / (FOG_END - FOG_START), 0.0, 1.0);

    return mix(fog_color, final_color, fog_factor);
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
        let world_pos = position_data.xyz;

        // Check if this pixel belongs to the current chunk
        // Determine which chunk owns this pixel by comparing world position ranges
        let grid_size_f = f32(context.grid_size);
        let chunk_min_x = f32(chunk_world_pos.x);
        let chunk_max_x = chunk_min_x + grid_size_f;
        let chunk_min_y = f32(chunk_world_pos.y);
        let chunk_max_y = chunk_min_y + grid_size_f;
        let chunk_min_z = f32(chunk_world_pos.z);
        let chunk_max_z = chunk_min_z + grid_size_f;

        let in_chunk_x = world_pos.x >= chunk_min_x && world_pos.x < chunk_max_x;
        let in_chunk_y = world_pos.y >= chunk_min_y && world_pos.y < chunk_max_y;
        let in_chunk_z = world_pos.z >= chunk_min_z && world_pos.z < chunk_max_z;

        // Only render if pixel belongs to current chunk
        if (in_chunk_x && in_chunk_y && in_chunk_z) {

            // Render geometry with lighting
            let normal_data = textureLoad(normalTexture, pixel_coord, 0);
            let diffuse_data = textureLoad(diffuseTexture, pixel_coord, 0);
            let distance = position_data.w;
            let world_normal = normalize(normal_data.xyz);
            let diffuse_color = diffuse_data.xyz;
            let light_visibility = diffuse_data.w;

            // Use baked vertex color with fog and environment
            let final_color = apply_fog_and_environment(diffuse_color, light_visibility, world_normal, camera_pos, world_pos, distance);
            return vec4<f32>(final_color, 1.0);
        }

        // Pixel belongs to a different chunk, will be rendered by that chunk's pass
        // Return transparent to preserve previous pass results
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Render background
    return calculate_background_color(uv, camera_pos);
}
