#import "perlin.wgsl"
#import "voronoi.wgsl"
#import "../data/context.wgsl"

// input
@group(1) @binding(0) var<uniform> context: Context;

// output
@group(0) @binding(0) var<storage, read_write> noise: array<f32>;

// Generate continuous SDF values instead of binary 0/1
fn generate_sdf_noise(pos: vec3<u32>) -> f32 {
    // Generate deterministic noise that doesn't depend on time for TAA stability
    let noise_value = rock_voronoi3(vec3<f32>(pos+100) / 600.0, 4,10);

    // Convert noise from [0,1] range to SDF values
    // Values > 0.5 become negative (inside), values < 0.5 become positive (outside)
    let threshold = 0.5;
    let sdf_range = 2.0; // Maximum distance value

    return (threshold - noise_value) * sdf_range;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    noise[to1D(id)] = generate_sdf_noise(id);
}
