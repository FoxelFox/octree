#import "../../data/context.wgsl"

// Constants
const FOG_START = 128.0;
const FOG_END = 512.0;

@group(0) @binding(0) var<uniform> context: Context;
@group(0) @binding(1) var<uniform> chunk_world_pos: vec3<i32>;
@group(1) @binding(0) var positionTexture: texture_2d<f32>;
@group(1) @binding(1) var normalTexture: texture_2d<f32>;
@group(1) @binding(2) var diffuseTexture: texture_2d<f32>;
@group(1) @binding(3) var depthTexture: texture_depth_2d;
@group(2) @binding(0) var space_background_texture: texture_2d<f32>;
@group(3) @binding(0) var<storage, read> light_data: array<vec2<f32>>;
// Neighbor light buffers (6 directions: -X, +X, -Y, +Y, -Z, +Z)
@group(3) @binding(1) var<storage, read> neighbor_light_nx: array<vec2<f32>>; // -X neighbor
@group(3) @binding(2) var<storage, read> neighbor_light_px: array<vec2<f32>>; // +X neighbor
@group(3) @binding(3) var<storage, read> neighbor_light_ny: array<vec2<f32>>; // -Y neighbor
@group(3) @binding(4) var<storage, read> neighbor_light_py: array<vec2<f32>>; // +Y neighbor
@group(3) @binding(5) var<storage, read> neighbor_light_nz: array<vec2<f32>>; // -Z neighbor
@group(3) @binding(6) var<storage, read> neighbor_light_pz: array<vec2<f32>>; // +Z neighbor

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

// Convert world position to chunk-local compressed grid coordinates
fn worldToCompressedGrid(world_pos: vec3<f32>) -> vec3<f32> {
    // Subtract chunk world position to get chunk-local coordinates
    let chunk_local_pos = world_pos - vec3<f32>(chunk_world_pos);
    return chunk_local_pos / COMPRESSION;
}

// Sample light data from the compressed grid using trilinear interpolation
fn sampleLightData(world_pos: vec3<f32>) -> vec2<f32> {
    let grid_pos = worldToCompressedGrid(world_pos);

    // Don't clamp - allow sampling from neighbor chunks via getLightDataAtGrid
    // Get integer coordinates for the 8 surrounding cells
    let base_pos = floor(grid_pos);
    let fract_pos = grid_pos - base_pos;

    // Sample 8 neighboring light values for trilinear interpolation
    // getLightDataAtGrid will handle out-of-bounds by sampling from neighbor buffers
    let p000 = getLightDataAtGrid(vec3<i32>(base_pos));
    let p001 = getLightDataAtGrid(vec3<i32>(base_pos + vec3<f32>(0.0, 0.0, 1.0)));
    let p010 = getLightDataAtGrid(vec3<i32>(base_pos + vec3<f32>(0.0, 1.0, 0.0)));
    let p011 = getLightDataAtGrid(vec3<i32>(base_pos + vec3<f32>(0.0, 1.0, 1.0)));
    let p100 = getLightDataAtGrid(vec3<i32>(base_pos + vec3<f32>(1.0, 0.0, 0.0)));
    let p101 = getLightDataAtGrid(vec3<i32>(base_pos + vec3<f32>(1.0, 0.0, 1.0)));
    let p110 = getLightDataAtGrid(vec3<i32>(base_pos + vec3<f32>(1.0, 1.0, 0.0)));
    let p111 = getLightDataAtGrid(vec3<i32>(base_pos + vec3<f32>(1.0, 1.0, 1.0)));

    // Trilinear interpolation
    let c00 = mix(p000, p100, fract_pos.x);
    let c01 = mix(p001, p101, fract_pos.x);
    let c10 = mix(p010, p110, fract_pos.x);
    let c11 = mix(p011, p111, fract_pos.x);

    let c0 = mix(c00, c10, fract_pos.y);
    let c1 = mix(c01, c11, fract_pos.y);

    return mix(c0, c1, fract_pos.z);
}

// Get light data at specific grid coordinates, including from neighbor chunks
fn getLightDataAtGrid(grid_pos: vec3<i32>) -> vec2<f32> {
    let size = i32(context.grid_size / COMPRESSION);

    // Check if we need to sample from a neighbor chunk
    if (grid_pos.x < 0) {
        // Sample from -X neighbor at their rightmost edge
        let neighbor_pos = vec3<i32>(size - 1, grid_pos.y, grid_pos.z);
        if (neighbor_pos.y >= 0 && neighbor_pos.y < size && neighbor_pos.z >= 0 && neighbor_pos.z < size) {
            let index = u32(neighbor_pos.z * size * size + neighbor_pos.y * size + neighbor_pos.x);
            return neighbor_light_nx[index];
        }
        return vec2<f32>(0.0, 1.0);
    } else if (grid_pos.x >= size) {
        // Sample from +X neighbor at their leftmost edge
        let neighbor_pos = vec3<i32>(0, grid_pos.y, grid_pos.z);
        if (neighbor_pos.y >= 0 && neighbor_pos.y < size && neighbor_pos.z >= 0 && neighbor_pos.z < size) {
            let index = u32(neighbor_pos.z * size * size + neighbor_pos.y * size + neighbor_pos.x);
            return neighbor_light_px[index];
        }
        return vec2<f32>(0.0, 1.0);
    } else if (grid_pos.y < 0) {
        // Sample from -Y neighbor
        let neighbor_pos = vec3<i32>(grid_pos.x, size - 1, grid_pos.z);
        if (neighbor_pos.x >= 0 && neighbor_pos.x < size && neighbor_pos.z >= 0 && neighbor_pos.z < size) {
            let index = u32(neighbor_pos.z * size * size + neighbor_pos.y * size + neighbor_pos.x);
            return neighbor_light_ny[index];
        }
        return vec2<f32>(0.0, 1.0);
    } else if (grid_pos.y >= size) {
        // Sample from +Y neighbor
        let neighbor_pos = vec3<i32>(grid_pos.x, 0, grid_pos.z);
        if (neighbor_pos.x >= 0 && neighbor_pos.x < size && neighbor_pos.z >= 0 && neighbor_pos.z < size) {
            let index = u32(neighbor_pos.z * size * size + neighbor_pos.y * size + neighbor_pos.x);
            return neighbor_light_py[index];
        }
        return vec2<f32>(0.0, 1.0);
    } else if (grid_pos.z < 0) {
        // Sample from -Z neighbor
        let neighbor_pos = vec3<i32>(grid_pos.x, grid_pos.y, size - 1);
        if (neighbor_pos.x >= 0 && neighbor_pos.x < size && neighbor_pos.y >= 0 && neighbor_pos.y < size) {
            let index = u32(neighbor_pos.z * size * size + neighbor_pos.y * size + neighbor_pos.x);
            return neighbor_light_nz[index];
        }
        return vec2<f32>(0.0, 1.0);
    } else if (grid_pos.z >= size) {
        // Sample from +Z neighbor
        let neighbor_pos = vec3<i32>(grid_pos.x, grid_pos.y, 0);
        if (neighbor_pos.x >= 0 && neighbor_pos.x < size && neighbor_pos.y >= 0 && neighbor_pos.y < size) {
            let index = u32(neighbor_pos.z * size * size + neighbor_pos.y * size + neighbor_pos.x);
            return neighbor_light_pz[index];
        }
        return vec2<f32>(0.0, 1.0);
    }

    // Sample from current chunk
    let index = u32(grid_pos.z * size * size + grid_pos.y * size + grid_pos.x);
    return light_data[index];
}

// Sample directional light from the floodfill data
fn sampleDirectionalLight(world_pos: vec3<f32>, direction: vec3<f32>, step_size: f32) -> f32 {
    // Sample light in the given direction to simulate directional lighting
    let sample_pos = world_pos + direction * step_size;
    let light_info = sampleLightData(sample_pos);
    return light_info.x; // Light intensity
}

// Calculate lighting for a given world position and surface properties
fn calculate_lighting(world_pos: vec3<f32>, world_normal: vec3<f32>, diffuse_color: vec3<f32>, camera_pos: vec3<f32>, distance: f32) -> vec3<f32> {
    // Sample voxel-based lighting data at surface
    let light_info = sampleLightData(world_pos);
    let base_light_intensity = light_info.x; // R channel: light intensity
    let shadow_factor = light_info.y;        // G channel: shadow factor

    // Create directional lighting effect by sampling light from different directions
    // This simulates how surfaces facing light sources should be brighter

    // Primary skylight direction (from above)
    let sky_dir = vec3<f32>(0.0, 1.0, 0.0);
    let sky_light = sampleDirectionalLight(world_pos, sky_dir, 4.0);
    let sky_contribution = max(dot(-world_normal, sky_dir), 0.0) * sky_light * 0.8;

    // Sample light from multiple directions for better normal response
    let directions = array<vec3<f32>, 6>(
        vec3<f32>( 1.0,  0.0,  0.0), // +X
        vec3<f32>(-1.0,  0.0,  0.0), // -X
        vec3<f32>( 0.0,  1.0,  0.0), // +Y (up)
        vec3<f32>( 0.0, -1.0,  0.0), // -Y (down)
        vec3<f32>( 0.0,  0.0,  1.0), // +Z
        vec3<f32>( 0.0,  0.0, -1.0)  // -Z
    );

    var directional_light = 0.0;
    for (var i = 0; i < 6; i++) {
        let dir_light = sampleDirectionalLight(world_pos, directions[i], 2.0);
        let normal_factor = max(dot(-world_normal, directions[i]), 0.0);
        directional_light += dir_light * normal_factor;
    }
    directional_light *= 0.15; // Scale contribution

    // Combine base ambient with directional effects
    let total_lighting = base_light_intensity * 0.6 + // Base ambient
                        sky_contribution +             // Sky directional
                        directional_light;             // Multi-directional

    // Preserve normal contrast while preventing over-brightening
    // Use a softer approach that maintains relative differences
    let exposure_adjustment = 1.0 / (1.0 + total_lighting * 0.3); // Soft exposure curve
    let contrast_preserved_lighting = total_lighting * exposure_adjustment;

    // Allow complete darkness in caves - no minimum lighting
    let clamped_lighting = clamp(contrast_preserved_lighting, 0.0, 1.1);

    // Apply lighting to surface color
    let lit_color = diffuse_color * clamped_lighting;

    // Add environment reflection (reduced in dark areas but still present)
    let env_reflection = sample_nebula_only(world_normal) * 0.08 * (1.0 - shadow_factor);
    let final_color = lit_color + diffuse_color * env_reflection;

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

            let lit_color = calculate_lighting(world_pos, world_normal, diffuse_color, camera_pos, distance);
            return vec4<f32>(lit_color, 1.0);
        }

        // Pixel belongs to a different chunk, will be rendered by that chunk's pass
        // Return transparent to preserve previous pass results
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Render background
    return calculate_background_color(uv, camera_pos);
}
