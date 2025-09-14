#import "voronoi.wgsl"

// Star Configuration Constants
const MAIN_STAR_DENSITY = 1200.0;
const MAIN_STAR_THRESHOLD = 0.015;
const MAIN_STAR_SHARPNESS = 0.003;
const MAIN_STAR_INTENSITY = 2.5;

const GIANT_STAR_DENSITY = 400.0;
const GIANT_STAR_THRESHOLD = 0.025;
const GIANT_STAR_SHARPNESS = 0.008;
const GIANT_STAR_INTENSITY = 3.0;

const HOT_STAR_DENSITY = 800.0;
const HOT_STAR_THRESHOLD = 0.1;
const HOT_STAR_SHARPNESS = 0.004;
const HOT_STAR_INTENSITY = 2.0;

const SUN_STAR_DENSITY = 1600.0;
const SUN_STAR_THRESHOLD = 0.1;
const SUN_STAR_SHARPNESS = 0.003;
const SUN_STAR_INTENSITY = 2.2;

const DWARF_STAR_DENSITY = 1400.0;
const DWARF_STAR_THRESHOLD = 0.1;
const DWARF_STAR_SHARPNESS = 0.005;
const DWARF_STAR_INTENSITY = 1.5;

const MICRO_STAR_DENSITY = 3500.0;
const MICRO_STAR_THRESHOLD = 0.1;
const MICRO_STAR_SHARPNESS = 0.001;
const MICRO_STAR_INTENSITY = 0.6;

const TINY_STAR_DENSITY = 5000.0;
const TINY_STAR_THRESHOLD = 0.1;
const TINY_STAR_SHARPNESS = 0.0005;
const TINY_STAR_INTENSITY = 0.4;

const DIM_STAR_DENSITY = 2000.0;
const DIM_STAR_THRESHOLD = 0.1;
const DIM_STAR_SHARPNESS = 0.02;
const DIM_STAR_INTENSITY = 0.8;

// Output texture
@group(0) @binding(0) var space_texture: texture_storage_2d<rgba8unorm, write>;

// Generate stars with different colors and densities
fn generate_stars(ray_dir: vec3<f32>) -> vec3<f32> {
    var star_color = vec3<f32>(0.0);

    // White/blue main stars (high density, bright)
    let main_pos = ray_dir * MAIN_STAR_DENSITY;
    let main_noise = voronoi3(main_pos);
    let main_intensity = 1.0 - smoothstep(MAIN_STAR_SHARPNESS, MAIN_STAR_THRESHOLD, main_noise);
    let main_stars = main_intensity * main_intensity * main_intensity;
    star_color += vec3<f32>(1.0, 0.95, 0.9) * main_stars * MAIN_STAR_INTENSITY;

    // Orange/red giant stars (lower density, larger)
    let giant_pos = ray_dir * GIANT_STAR_DENSITY + vec3<f32>(100.0, 50.0, 200.0);
    let giant_noise = voronoi3(giant_pos);
    let giant_intensity = 1.0 - smoothstep(GIANT_STAR_SHARPNESS, GIANT_STAR_THRESHOLD, giant_noise);
    let giant_stars = giant_intensity * giant_intensity;
    star_color += vec3<f32>(1.0, 0.6, 0.3) * giant_stars * GIANT_STAR_INTENSITY;

    // Blue/white hot stars (medium density)
    let hot_pos = ray_dir * HOT_STAR_DENSITY + vec3<f32>(300.0, 150.0, 75.0);
    let hot_noise = voronoi3(hot_pos);
    let hot_intensity = 1.0 - smoothstep(HOT_STAR_SHARPNESS, HOT_STAR_THRESHOLD, hot_noise);
    let hot_stars = hot_intensity * hot_intensity * hot_intensity;
    star_color += vec3<f32>(0.8, 0.9, 1.0) * hot_stars * HOT_STAR_INTENSITY;

    // Yellow/white sun-like stars
    let sun_pos = ray_dir * SUN_STAR_DENSITY + vec3<f32>(700.0, 400.0, 300.0);
    let sun_noise = voronoi3(sun_pos);
    let sun_intensity = 1.0 - smoothstep(SUN_STAR_SHARPNESS, SUN_STAR_THRESHOLD, sun_noise);
    let sun_stars = sun_intensity * sun_intensity * sun_intensity;
    star_color += vec3<f32>(1.0, 0.8, 0.5) * sun_stars * SUN_STAR_INTENSITY;

    // Red dwarf stars (medium density, dim red)
    let dwarf_pos = ray_dir * DWARF_STAR_DENSITY + vec3<f32>(900.0, 200.0, 600.0);
    let dwarf_noise = voronoi3(dwarf_pos);
    let dwarf_intensity = 1.0 - smoothstep(DWARF_STAR_SHARPNESS, DWARF_STAR_THRESHOLD, dwarf_noise);
    let dwarf_stars = dwarf_intensity * dwarf_intensity;
    star_color += vec3<f32>(1.0, 0.4, 0.2) * dwarf_stars * DWARF_STAR_INTENSITY;

    // Very small distant stars (ultra high density, tiny)
    let micro_pos = ray_dir * MICRO_STAR_DENSITY + vec3<f32>(1200.0, 800.0, 400.0);
    let micro_noise = voronoi3(micro_pos);
    let micro_intensity = 1.0 - smoothstep(MICRO_STAR_SHARPNESS, MICRO_STAR_THRESHOLD, micro_noise);
    let micro_stars = micro_intensity * micro_intensity;
    star_color += vec3<f32>(0.8, 0.8, 0.9) * micro_stars * MICRO_STAR_INTENSITY;

    // Tiny pinprick stars (extreme density, barely visible)
    let tiny_pos = ray_dir * TINY_STAR_DENSITY + vec3<f32>(1500.0, 1000.0, 800.0);
    let tiny_noise = voronoi3(tiny_pos);
    let tiny_intensity = 1.0 - smoothstep(TINY_STAR_SHARPNESS, TINY_STAR_THRESHOLD, tiny_noise);
    let tiny_stars = tiny_intensity * tiny_intensity;
    star_color += vec3<f32>(0.7, 0.75, 0.8) * tiny_stars * TINY_STAR_INTENSITY;

    // Distant dim stars (very high density, very dim)
    let dim_pos = ray_dir * DIM_STAR_DENSITY + vec3<f32>(500.0, 300.0, 100.0);
    let dim_noise = voronoi3(dim_pos);
    let dim_intensity = 1.0 - smoothstep(DIM_STAR_SHARPNESS, DIM_STAR_THRESHOLD, dim_noise);
    let dim_stars = dim_intensity * dim_intensity;
    star_color += vec3<f32>(0.7, 0.7, 0.8) * dim_stars * DIM_STAR_INTENSITY;

    return star_color;
}

// Generate nebula using fractal voronoi
fn generate_nebula(ray_dir: vec3<f32>) -> vec3<f32> {
    let nebula_scale = 2.0;
    let nebula_pos = ray_dir * nebula_scale;

    // Multiple layers of nebula
    let nebula1 = fractal_voronoi3(nebula_pos * 0.5, 6u) * 0.6;
    let nebula2 = fractal_voronoi3(nebula_pos * 1.2, 4u) * 0.4;
    let nebula3 = voronoi3(nebula_pos * 3.0) * 0.2;

    let combined_nebula = nebula1 + nebula2 + nebula3;

    // Space nebula colors - keeping similar to original but more space-like
    let base_color = abs(ray_dir) * 0.5 - 0.1; // Darker base
    let nebula_color = base_color + vec3<f32>(1.0, 1.0, 1.0) * combined_nebula;

    return nebula_color;
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

    // Generate space background
    let stars = generate_stars(ray_dir);
    let nebula = generate_nebula(ray_dir);

    // Store stars in alpha channel, nebula in RGB
    let space_color = vec4<f32>(nebula, length(stars));

    // Write to texture
    textureStore(space_texture, coord, space_color);
}
