// Blue noise generation compute shader
// Generates a small tileable blue noise texture

@group(0) @binding(0) var blueNoiseTexture: texture_storage_2d<rgba8unorm, write>;

const BLUE_NOISE_SIZE: u32 = 64u;

// Simple blue noise generation using Mitchell's best-candidate algorithm
// This creates a Poisson distribution in spectral domain
fn hash2(p: vec2<u32>) -> f32 {
    var p1 = p.x;
    var p2 = p.y;
    p1 = (p1 ^ 61u) ^ (p1 >> 16u);
    p1 = p1 + (p1 << 3u);
    p1 = p1 ^ (p1 >> 4u);
    p1 = p1 * 0x27d4eb2du;
    p1 = p1 ^ (p1 >> 15u);
    
    p2 = (p2 ^ 61u) ^ (p2 >> 16u);
    p2 = p2 + (p2 << 3u);
    p2 = p2 ^ (p2 >> 4u);
    p2 = p2 * 0x27d4eb2du;
    p2 = p2 ^ (p2 >> 15u);
    
    return fract(f32((p1 ^ p2) * 0x27d4eb2du) / 4294967296.0);
}

// Generate blue noise value using spectral distribution
fn generateBlueNoise(coord: vec2<u32>) -> f32 {
    let scale = 1.618033988749; // Golden ratio for better distribution
    var result = 0.0;
    var freq = 1.0;
    var amp = 1.0;
    
    // Multiple octaves with decreasing amplitude
    for (var i = 0u; i < 4u; i++) {
        let scaled_coord = vec2<u32>(vec2<f32>(coord) * freq);
        let noise = hash2(scaled_coord + vec2<u32>(i * 73u, i * 157u));
        
        // Blue noise has higher frequency content
        // Weight higher frequencies more heavily
        result += noise * amp * (1.0 + f32(i) * 0.5);
        freq *= scale;
        amp *= 0.6;
    }
    
    // Normalize and add slight bias to avoid pure black/white
    return fract(result * 0.347) * 0.9 + 0.05;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coord = global_id.xy;
    
    if (coord.x >= BLUE_NOISE_SIZE || coord.y >= BLUE_NOISE_SIZE) {
        return;
    }
    
    // Generate blue noise value
    let noise_value = generateBlueNoise(coord);
    
    // Store as grayscale in all channels for easy sampling
    let color = vec4<f32>(noise_value, noise_value, noise_value, 1.0);
    
    textureStore(blueNoiseTexture, vec2<i32>(coord), color);
}