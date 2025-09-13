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
    let pos_f = vec3<f32>(pos);

    // Generate rocky surface base using fractal noise
    let surface_noise = rock_voronoi3(pos_f / 80.0, 2, 10);
    let surface_detail = rock_voronoi3(pos_f / 100.0, 8, 10) * 0.3;
    let rocky_surface = surface_noise + surface_detail;

    // Create a base rocky terrain with height variation
    let terrain_height = 64.0; // Base terrain level
    let height_variation = 32.0; // How much the terrain varies
    let base_terrain = pos_f.y - (terrain_height + rocky_surface * height_variation);

    // Generate cave systems using 3D Voronoi noise
    let cave_scale = 200.0; // Size of cave chambers
    let cave_noise1 = rock_voronoi3(pos_f / cave_scale, 3, 5);
    let cave_noise2 = rock_voronoi3((pos_f + vec3<f32>(1000.0)) / (cave_scale * 0.7), 2, 4);
    let cave_noise3 = rock_voronoi3((pos_f + vec3<f32>(2000.0)) / (cave_scale * 1.3), 4, 3);
    
    // Create cave chambers - combine multiple noise layers for complex cave shapes
    let cave_chambers = max(max(cave_noise1 - 0.4, cave_noise2 - 0.45), cave_noise3 - 0.5);
    
    // Create cave tunnels using different noise
    let tunnel_noise1 = rock_voronoi3(pos_f / 150.0 + vec3<f32>(500.0), 2, 6);
    let tunnel_noise2 = rock_voronoi3(pos_f / 180.0 + vec3<f32>(1500.0), 3, 4);
    let cave_tunnels = max(tunnel_noise1 - 0.6, tunnel_noise2 - 0.65);
    
    // Combine chambers and tunnels
    let cave_system = max(cave_chambers, cave_tunnels);
    
    // Create cave SDF - positive values are inside caves (air)
    let cave_sdf = cave_system * 20.0; // Scale up the cave effect

    // Generate stones that sit on top of the rocky surface
    let stone_noise = rock_voronoi3((pos_f + vec3<f32>(100.0)) / 300.0, 6, 8);
    let stone_size = 8.0; // Size of individual stones
    let stone_threshold = 0.6; // How sparse the stones are

    // Only place stones above the base terrain
    var stone_sdf = 1000.0; // Far away by default
    if (base_terrain < 0.0 && stone_noise > stone_threshold) {
        // Distance from stone center
        let stone_center_noise = rock_voronoi3(pos_f / 150.0, 4, 3);
        stone_sdf = length(fract(pos_f / stone_size) - 0.5) * stone_size - (stone_size * 0.3);
        // Add some variation to stone shape
        stone_sdf += stone_center_noise * 2.0;
    }

    // Combine terrain and stones using min operation (union in SDF)
    var final_sdf = min(base_terrain, stone_sdf);
    
    // Subtract caves from the terrain (max operation with negative cave SDF)
    final_sdf = max(final_sdf, cave_sdf);

    // Clamp final SDF to 0-1 range for voxel buffer
    return clamp(final_sdf, -1.0, 1.0);
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
    let color = 0xDDDDDDu;

    voxels[to1D(id)] = VoxelData(density, color);
}
