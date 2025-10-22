#import "voronoi.wgsl"

// Atmosphere Configuration Constants
const HORIZON_HEIGHT = 0.0; // Y-coordinate where horizon starts
const SKY_TOP_COLOR = vec3<f32>(0.7, 0.7, 0.8); // Dark blue at top
const SKY_MIDDLE_COLOR = vec3<f32>(0.8, 0.5, 0.3); // Lighter blue in middle
const HORIZON_COLOR = vec3<f32>(0.9, 0.7, 0.5); // Warm orange/yellow at horizon
const HORIZON_GLOW_COLOR = vec3<f32>(0.95, 0.75, 0.55); // Softer glow near horizon
const HORIZON_INTENSITY = 0.3;
const SKY_GRADIENT_POWER = 2.0;
const HORIZON_THICKNESS = 0.15;

// Output texture
@group(0) @binding(0) var space_texture: texture_storage_2d<rgba8unorm, write>;

// Generate atmospheric horizon gradient
fn generate_atmosphere(ray_dir: vec3<f32>) -> vec3<f32> {
    // Normalize the ray direction
    let dir = normalize(ray_dir);

    // Calculate height factor (y component ranges from -1 to 1)
    // Map it to 0 (bottom) to 1 (top)
    let height = (dir.y + 1.0) * 0.5;

    // Create fully smooth continuous gradient with no hard edges
    let height_smoothed = smoothstep(0.0, 1.0, height);

    // Blend between three colors using smooth interpolation
    // Lower half: horizon to middle
    let lower_blend = smoothstep(0.0, 0.5, height_smoothed);
    let lower_color = mix(HORIZON_COLOR, SKY_MIDDLE_COLOR, lower_blend);

    // Upper half: middle to top
    let upper_blend = smoothstep(0.5, 1.0, height_smoothed);
    let upper_color = mix(SKY_MIDDLE_COLOR, SKY_TOP_COLOR, upper_blend);

    // Smoothly transition between lower and upper
    let transition = smoothstep(0.4, 0.6, height_smoothed);
    var sky_color = mix(lower_color, upper_color, transition);

    // Add subtle horizon glow that blends smoothly
    let horizon_distance = abs(dir.y);
    let horizon_glow = exp(-horizon_distance * 3.0) * HORIZON_INTENSITY;
    let glow_blend = smoothstep(0.0, 0.4, horizon_distance);
    sky_color = mix(sky_color + HORIZON_GLOW_COLOR * horizon_glow, sky_color, glow_blend);

    // Add subtle atmospheric noise using voronoi
    let noise_scale = 4.0;
    let noise_pos = dir * noise_scale;
    let noise = fractal_voronoi3(noise_pos, 3u) * 0.08;

    // Add subtle cloud-like variations
    let cloud_scale = 2.0;
    let cloud_pos = dir * cloud_scale + vec3<f32>(100.0, 50.0, 0.0);
    let clouds = fractal_voronoi3(cloud_pos, 4u) * 0.12;

    // Apply atmospheric effects only in upper atmosphere
    let atmospheric_effect = max(0.0, height - 0.3) * (noise + clouds);
    sky_color += sky_color * atmospheric_effect;

    return sky_color;
}

// Convert UV to ray direction for cubemap-like sampling
fn uv_to_ray_direction(uv: vec2<f32>) -> vec3<f32> {
    // Convert UV to spherical coordinates
    let theta = uv.x * 2.0 * 3.14159265; // Azimuth: 0 to 2π
    let phi = uv.y * 3.14159265; // Elevation: 0 to π

    // Convert spherical to cartesian
    let sin_phi = sin(phi);
    return vec3<f32>(
        sin_phi * cos(theta),
        cos(phi),
        sin_phi * sin(theta)
    );
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let texture_size = textureDimensions(space_texture);
    let coord = vec2<i32>(global_id.xy);

    // Skip if out of bounds
    if (coord.x >= i32(texture_size.x) || coord.y >= i32(texture_size.y)) {
        return;
    }

    // Convert pixel coordinate to UV
    let uv = vec2<f32>(coord) / vec2<f32>(texture_size);

    // Get ray direction from UV
    let ray_dir = uv_to_ray_direction(uv);

    // Generate atmospheric horizon background
    let atmosphere = generate_atmosphere(ray_dir);

    // Store atmosphere in RGB, alpha = 0 (no stars)
    let background_color = vec4<f32>(atmosphere, 0.0);

    // Write to texture
    textureStore(space_texture, coord, background_color);
}
