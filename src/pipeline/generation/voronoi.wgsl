// Hash function for generating pseudo-random values
fn hash(n: f32) -> f32 {
    return fract(sin(n) * 43758.5453123);
}

// 3D hash function
fn hash3(p: vec3f) -> vec3f {
    let q = vec3f(
        dot(p, vec3f(127.1, 311.7, 74.7)),
        dot(p, vec3f(269.5, 183.3, 246.1)),
        dot(p, vec3f(113.5, 271.9, 124.6))
    );
    return fract(sin(q) * 43758.5453123);
}

// 3D Voronoi noise - returns distance to nearest feature point
fn voronoi3(p: vec3f) -> f32 {
    let i = floor(p);
    let f = fract(p);
    
    var min_dist = 1.0;
    
    // Check 27 neighboring cells (3x3x3)
    for (var z = -1; z <= 1; z++) {
        for (var y = -1; y <= 1; y++) {
            for (var x = -1; x <= 1; x++) {
                let neighbor = vec3f(f32(x), f32(y), f32(z));
                let point = hash3(i + neighbor);
                let diff = neighbor + point - f;
                let dist = dot(diff, diff);
                min_dist = min(min_dist, dist);
            }
        }
    }
    
    return sqrt(min_dist);
}

// Enhanced voronoi with multiple features for rock-like structures
fn voronoi_rock3(p: vec3f) -> f32 {
    let i = floor(p);
    let f = fract(p);
    
    var min_dist1 = 8.0;
    var min_dist2 = 8.0;
    
    // Check 27 neighboring cells
    for (var z = -1; z <= 1; z++) {
        for (var y = -1; y <= 1; y++) {
            for (var x = -1; x <= 1; x++) {
                let neighbor = vec3f(f32(x), f32(y), f32(z));
                let point = hash3(i + neighbor);
                let diff = neighbor + point - f;
                let dist = dot(diff, diff);
                
                if (dist < min_dist1) {
                    min_dist2 = min_dist1;
                    min_dist1 = dist;
                } else if (dist < min_dist2) {
                    min_dist2 = dist;
                }
            }
        }
    }
    
    // Return difference between first and second closest points
    // This creates more interesting cellular patterns
    return sqrt(min_dist2) - sqrt(min_dist1);
}

// Fractal voronoi for complex rock structures
fn fractal_voronoi3(p: vec3f, octaves: u32) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;
    var pos = p;
    
    for (var i = 0u; i < octaves; i++) {
        value += voronoi_rock3(pos * frequency) * amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    
    return value;
}

// Main voronoi function with configurable parameters for rock generation
fn rock_voronoi3(p: vec3f, scale: f32, detail_octaves: u32) -> f32 {
    let scaled_pos = p * scale;
    
    // Base voronoi structure
    let base_voronoi = voronoi_rock3(scaled_pos);
    
    // Add fractal detail for more complex rock-like features
    let detail = fractal_voronoi3(scaled_pos * 2.0, detail_octaves) * 0.3;
    
    // Combine base structure with detail
    return base_voronoi + detail;
}

// Simple voronoi noise compatible with existing noise interface
fn voronoi_noise3(p: vec3f) -> f32 {
    return rock_voronoi3(p, 1.0, 3u);
}