#import "perlin.wgsl"
#import "voronoi.wgsl"
#import "../data/context.wgsl"

// input
@group(1) @binding(0) var<uniform> context: Context;

// Voxel data structure containing density and color
struct VoxelData {
    density: f32,
    color: u32, // Packed RGBA color (8 bits per channel)
}

// output
@group(0) @binding(0) var<storage, read_write> voxels: array<VoxelData>;

// Generate continuous SDF values instead of binary 0/1
fn generate_sdf_noise(pos: vec3<u32>) -> f32 {
    // Generate deterministic noise that doesn't depend on time for TAA stability
    let noise_value = rock_voronoi3(vec3<f32>(pos+10) / 500.0, 4,10);

    // Convert noise from [0,1] range to SDF values
    // Values > 0.5 become negative (inside), values < 0.5 become positive (outside)
    let threshold = 0.5;
    let sdf_range = 2.0; // Maximum distance value

    return (threshold - noise_value) * sdf_range;
}

// Generate candy-colored voxels
fn generate_color(pos: vec3<u32>, density: f32) -> u32 {
    let pos_f = vec3<f32>(pos);

    // Use multiple noise sources to create different candy zones
    let color_noise1 = rock_voronoi3(pos_f / 1000.0, 1, 2);
    let color_noise2 = rock_voronoi3(pos_f / 1500.0 + vec3(100.0), 2, 2);
    let color_noise3 = rock_voronoi3(pos_f / 800.0 + vec3(200.0), 3, 2);
    let detail_noise = rock_voronoi3(pos_f / 400.0 + vec3(300.0), 4, 2);

    // Create striped/layered candy patterns
    let stripe_pattern = sin(pos_f.y * 0.1) * 0.5 + 0.5;
    let swirl_pattern = sin(pos_f.x * 0.08 + pos_f.z * 0.12) * 0.5 + 0.5;

    var base_color: vec3<f32>;

    // Determine main candy color based on noise
    let color_selector = color_noise1 + color_noise2 * 0.5;

    if (color_selector < 0.15) {
        // Hot pink / magenta candy
        base_color = mix(vec3(1.0, 0.2, 0.8), vec3(1.0, 0.6, 0.9), stripe_pattern);
    } else if (color_selector < 0.3) {
        // Electric blue candy
        base_color = mix(vec3(0.0, 0.8, 1.0), vec3(0.4, 0.9, 1.0), stripe_pattern);
    } else if (color_selector < 0.45) {
        // Lime green candy
        base_color = mix(vec3(0.5, 1.0, 0.2), vec3(0.7, 1.0, 0.5), stripe_pattern);
    } else if (color_selector < 0.6) {
        // Orange creamsicle
        base_color = mix(vec3(1.0, 0.5, 0.1), vec3(1.0, 0.8, 0.3), stripe_pattern);
    } else if (color_selector < 0.75) {
        // Purple grape candy
        base_color = mix(vec3(0.6, 0.2, 1.0), vec3(0.8, 0.5, 1.0), stripe_pattern);
    } else if (color_selector < 0.9) {
        // Yellow lemon candy
        base_color = mix(vec3(1.0, 1.0, 0.2), vec3(1.0, 1.0, 0.6), stripe_pattern);
    } else {
        // Red cherry candy
        base_color = mix(vec3(1.0, 0.2, 0.3), vec3(1.0, 0.5, 0.6), stripe_pattern);
    }

    // Add candy swirl effects
    let swirl_intensity = color_noise3 * swirl_pattern;
    if (swirl_intensity > 0.7) {
        // White candy swirls
        base_color = mix(base_color, vec3(1.0, 0.95, 1.0), 0.6);
    } else if (swirl_intensity > 0.5) {
        // Pastel candy mixing
        let pastel_color = base_color * 0.7 + vec3(0.3, 0.3, 0.3);
        base_color = mix(base_color, pastel_color, 0.4);
    }

    // Add rainbow zones
    let rainbow_noise = rock_voronoi3(pos_f / 200.0 + vec3(400.0), 2, 7);
    if (rainbow_noise > 0.8) {
        // Rainbow candy zones - cycle through hues
        let hue = fract(pos_f.x * 0.01 + pos_f.z * 0.015 + rainbow_noise);
        let rainbow_color = hsv_to_rgb(vec3(hue, 0.9, 1.0));
        base_color = mix(base_color, rainbow_color, 0.5);
    }

    // Add sparkly highlights
    let sparkle_noise = rock_voronoi3(pos_f / 25.0 + vec3(500.0), 7, 2);
    if (sparkle_noise > 0.85) {
        // Bright white sparkles
        base_color = mix(base_color, vec3(1.0, 1.0, 1.0), 0.8);
    } else if (sparkle_noise > 0.75) {
        // Colored sparkles
        base_color = base_color * 1.3;
    }

    // Add surface detail variation
    let surface_variation = detail_noise - 0.5;
    base_color = clamp(base_color + vec3(surface_variation) * 0.2, vec3(0.0), vec3(1.0));

    // Make sure colors are vibrant
    base_color = clamp(base_color, vec3(0.0), vec3(1.0));

    // Convert to packed RGBA (with full alpha)
    let r = u32(base_color.r * 255.0) & 0xFFu;
    let g = u32(base_color.g * 255.0) & 0xFFu;
    let b = u32(base_color.b * 255.0) & 0xFFu;
    let a = 255u;

    return (a << 24u) | (b << 16u) | (g << 8u) | r;
}

// Convert HSV to RGB for rainbow effects
fn hsv_to_rgb(hsv: vec3<f32>) -> vec3<f32> {
    let h = hsv.x * 6.0;
    let s = hsv.y;
    let v = hsv.z;

    let c = v * s;
    let x = c * (1.0 - abs(fract(h * 0.5) * 2.0 - 1.0));
    let m = v - c;

    var rgb: vec3<f32>;
    if (h < 1.0) {
        rgb = vec3(c, x, 0.0);
    } else if (h < 2.0) {
        rgb = vec3(x, c, 0.0);
    } else if (h < 3.0) {
        rgb = vec3(0.0, c, x);
    } else if (h < 4.0) {
        rgb = vec3(0.0, x, c);
    } else if (h < 5.0) {
        rgb = vec3(x, 0.0, c);
    } else {
        rgb = vec3(c, 0.0, x);
    }

    return rgb + vec3(m);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let density = generate_sdf_noise(id);
    //let color = generate_color(id, density);
    let color = 0xFFFFFFFFu;

    voxels[to1D(id)] = VoxelData(density, color);
}
