// Internal noise generation functions - not exported to WASM

// ============================================================================
// Hash Functions (from voronoi.wgsl)
// ============================================================================

/// Hash function for generating pseudo-random values
fn hash(n: f32) -> f32 {
    (n.sin() * 43758.5453123).fract()
}

/// 3D hash function
fn hash3(p: [f32; 3]) -> [f32; 3] {
    let q = [
        p[0] * 127.1 + p[1] * 311.7 + p[2] * 74.7,
        p[0] * 269.5 + p[1] * 183.3 + p[2] * 246.1,
        p[0] * 113.5 + p[1] * 271.9 + p[2] * 124.6,
    ];
    [
        (q[0].sin() * 43758.5453123).fract(),
        (q[1].sin() * 43758.5453123).fract(),
        (q[2].sin() * 43758.5453123).fract(),
    ]
}

// ============================================================================
// Voronoi Noise Functions (from voronoi.wgsl)
// ============================================================================

/// 3D Voronoi noise - returns distance to nearest feature point
fn voronoi3(p: [f32; 3]) -> f32 {
    let i = [p[0].floor(), p[1].floor(), p[2].floor()];
    let f = [p[0].fract(), p[1].fract(), p[2].fract()];

    let mut min_dist: f32 = 1.0;

    // Check 27 neighboring cells (3x3x3)
    for z in -1..=1 {
        for y in -1..=1 {
            for x in -1..=1 {
                let neighbor = [x as f32, y as f32, z as f32];
                let point = hash3([i[0] + neighbor[0], i[1] + neighbor[1], i[2] + neighbor[2]]);
                let diff = [
                    neighbor[0] + point[0] - f[0],
                    neighbor[1] + point[1] - f[1],
                    neighbor[2] + point[2] - f[2],
                ];
                let dist = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
                min_dist = min_dist.min(dist);
            }
        }
    }

    min_dist.sqrt()
}

/// Enhanced voronoi with multiple features for rock-like structures
fn voronoi_rock3(p: [f32; 3]) -> f32 {
    let i = [p[0].floor(), p[1].floor(), p[2].floor()];
    let f = [p[0].fract(), p[1].fract(), p[2].fract()];

    let mut min_dist1 = 8.0;
    let mut min_dist2 = 8.0;

    // Check 27 neighboring cells
    for z in -1..=1 {
        for y in -1..=1 {
            for x in -1..=1 {
                let neighbor = [x as f32, y as f32, z as f32];
                let point = hash3([i[0] + neighbor[0], i[1] + neighbor[1], i[2] + neighbor[2]]);
                let diff = [
                    neighbor[0] + point[0] - f[0],
                    neighbor[1] + point[1] - f[1],
                    neighbor[2] + point[2] - f[2],
                ];
                let dist = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];

                if dist < min_dist1 {
                    min_dist2 = min_dist1;
                    min_dist1 = dist;
                } else if dist < min_dist2 {
                    min_dist2 = dist;
                }
            }
        }
    }

    // Return difference between first and second closest points
    min_dist2.sqrt() - min_dist1.sqrt()
}

/// Fractal voronoi for complex rock structures
fn fractal_voronoi3(p: [f32; 3], octaves: u32) -> f32 {
    let mut value = 0.0;
    let mut amplitude = 0.5;
    let mut frequency = 1.0;

    for _ in 0..octaves {
        let pos = [p[0] * frequency, p[1] * frequency, p[2] * frequency];
        value += voronoi_rock3(pos) * amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }

    value
}

/// Main voronoi function with configurable parameters for rock generation
fn rock_voronoi3(p: [f32; 3], scale: f32, detail_octaves: u32) -> f32 {
    let scaled_pos = [p[0] * scale, p[1] * scale, p[2] * scale];

    // Base voronoi structure
    let base_voronoi = voronoi_rock3(scaled_pos);

    // Add fractal detail for more complex rock-like features
    let detail_pos = [
        scaled_pos[0] * 2.0,
        scaled_pos[1] * 2.0,
        scaled_pos[2] * 2.0,
    ];
    let detail = fractal_voronoi3(detail_pos, detail_octaves) * 0.3;

    // Combine base structure with detail
    base_voronoi + detail
}

// ============================================================================
// SDF Noise Generation (from noise.wgsl)
// ============================================================================

/// Generate continuous SDF values instead of binary 0/1
pub(crate) fn generate_sdf_noise(pos: [u32; 3], chunk_offset: [i32; 3]) -> f32 {
    // Convert to world-space coordinates by adding chunk offset
    let world_pos = [
        pos[0] as f32 + chunk_offset[0] as f32,
        pos[1] as f32 + chunk_offset[1] as f32,
        pos[2] as f32 + chunk_offset[2] as f32,
    ];

    // Generate rocky surface base using fractal noise
    let surface_noise = rock_voronoi3(
        [
            world_pos[0] / 80.0,
            world_pos[1] / 80.0,
            world_pos[2] / 80.0,
        ],
        2.0,
        10,
    );

    let surface_detail = rock_voronoi3(
        [
            world_pos[0] / 100.0,
            world_pos[1] / 100.0,
            world_pos[2] / 100.0,
        ],
        8.0,
        10,
    ) * 0.3;
    let rocky_surface = surface_noise + surface_detail;

    // Create a base rocky terrain with height variation
    let terrain_height = 64.0;
    let height_variation = 32.0;
    let base_terrain = world_pos[1] - (terrain_height + rocky_surface * height_variation);

    // Generate cave systems using 3D Voronoi noise
    let cave_scale = 200.0;
    let cave_noise1 = rock_voronoi3(
        [
            world_pos[0] / cave_scale,
            world_pos[1] / cave_scale,
            world_pos[2] / cave_scale,
        ],
        3.0,
        5,
    );
    let cave_noise2 = rock_voronoi3(
        [
            (world_pos[0] + 1000.0) / (cave_scale * 0.7),
            (world_pos[1] + 1000.0) / (cave_scale * 0.7),
            (world_pos[2] + 1000.0) / (cave_scale * 0.7),
        ],
        2.0,
        4,
    );
    let cave_noise3 = rock_voronoi3(
        [
            (world_pos[0] + 2000.0) / (cave_scale * 1.3),
            (world_pos[1] + 2000.0) / (cave_scale * 1.3),
            (world_pos[2] + 2000.0) / (cave_scale * 1.3),
        ],
        4.0,
        3,
    );

    // Create cave chambers
    let cave_chambers = (cave_noise1 - 0.4)
        .max(cave_noise2 - 0.45)
        .max(cave_noise3 - 0.5);

    // Create cave tunnels
    let tunnel_noise1 = rock_voronoi3(
        [
            (world_pos[0] / 150.0) + 500.0,
            (world_pos[1] / 150.0) + 500.0,
            (world_pos[2] / 150.0) + 500.0,
        ],
        2.0,
        6,
    );
    let tunnel_noise2 = rock_voronoi3(
        [
            (world_pos[0] / 180.0) + 1500.0,
            (world_pos[1] / 180.0) + 1500.0,
            (world_pos[2] / 180.0) + 1500.0,
        ],
        3.0,
        4,
    );
    let cave_tunnels = (tunnel_noise1 - 0.6).max(tunnel_noise2 - 0.65);

    // Combine chambers and tunnels
    let cave_system = cave_chambers.max(cave_tunnels);

    // Create cave SDF
    let cave_sdf = cave_system * 20.0;

    // Generate stones
    let stone_noise = rock_voronoi3(
        [
            (world_pos[0] + 100.0) / 300.0,
            (world_pos[1] + 100.0) / 300.0,
            (world_pos[2] + 100.0) / 300.0,
        ],
        6.0,
        8,
    );
    let stone_size = 8.0;
    let stone_threshold = 0.6;

    // Only place stones above the base terrain
    let mut stone_sdf = 1000.0;
    if base_terrain < 0.0 && stone_noise > stone_threshold {
        let stone_center_noise = rock_voronoi3(
            [
                world_pos[0] / 150.0,
                world_pos[1] / 150.0,
                world_pos[2] / 150.0,
            ],
            4.0,
            3,
        );
        let fract_pos = [
            (world_pos[0] / stone_size).fract(),
            (world_pos[1] / stone_size).fract(),
            (world_pos[2] / stone_size).fract(),
        ];
        let diff = [fract_pos[0] - 0.5, fract_pos[1] - 0.5, fract_pos[2] - 0.5];
        let len = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]).sqrt();
        stone_sdf = len * stone_size - (stone_size * 0.3);
        stone_sdf += stone_center_noise * 2.0;
    }

    // Combine terrain and stones
    let mut final_sdf = base_terrain.min(stone_sdf);

    // Subtract caves from terrain
    final_sdf = final_sdf.max(cave_sdf);

    // Clamp final SDF to -1.0 to 1.0 range
    final_sdf.clamp(-1.0, 1.0)
}

// ============================================================================
// Color Generation (from noise.wgsl)
// ============================================================================

/// Convert HSV to RGB for rainbow effects
fn hsv_to_rgb(hsv: [f32; 3]) -> [f32; 3] {
    let h = hsv[0] * 6.0;
    let s = hsv[1];
    let v = hsv[2];

    let c = v * s;
    let x = c * (1.0 - ((h * 0.5).fract() * 2.0 - 1.0).abs());
    let m = v - c;

    let rgb = if h < 1.0 {
        [c, x, 0.0]
    } else if h < 2.0 {
        [x, c, 0.0]
    } else if h < 3.0 {
        [0.0, c, x]
    } else if h < 4.0 {
        [0.0, x, c]
    } else if h < 5.0 {
        [x, 0.0, c]
    } else {
        [c, 0.0, x]
    };

    [rgb[0] + m, rgb[1] + m, rgb[2] + m]
}

/// Mix two colors
fn mix_color(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] * (1.0 - t) + b[0] * t,
        a[1] * (1.0 - t) + b[1] * t,
        a[2] * (1.0 - t) + b[2] * t,
    ]
}

/// Generate candy-colored voxels
pub(crate) fn generate_color(pos: [u32; 3], chunk_offset: [i32; 3], _density: f32) -> u32 {
    // Convert to world-space coordinates
    let world_pos = [
        pos[0] as f32 + chunk_offset[0] as f32,
        pos[1] as f32 + chunk_offset[1] as f32,
        pos[2] as f32 + chunk_offset[2] as f32,
    ];

    // Use multiple noise sources to create different candy zones
    let color_noise1 = rock_voronoi3(
        [
            world_pos[0] / 1000.0,
            world_pos[1] / 1000.0,
            world_pos[2] / 1000.0,
        ],
        1.0,
        2,
    );
    let color_noise2 = rock_voronoi3(
        [
            (world_pos[0] + 100.0) / 1500.0,
            (world_pos[1] + 100.0) / 1500.0,
            (world_pos[2] + 100.0) / 1500.0,
        ],
        2.0,
        2,
    );
    let color_noise3 = rock_voronoi3(
        [
            (world_pos[0] + 200.0) / 800.0,
            (world_pos[1] + 200.0) / 800.0,
            (world_pos[2] + 200.0) / 800.0,
        ],
        3.0,
        2,
    );
    let detail_noise = rock_voronoi3(
        [
            (world_pos[0] + 300.0) / 400.0,
            (world_pos[1] + 300.0) / 400.0,
            (world_pos[2] + 300.0) / 400.0,
        ],
        4.0,
        2,
    );

    // Create striped/layered candy patterns
    let stripe_pattern = (world_pos[1] * 0.1).sin() * 0.5 + 0.5;
    let swirl_pattern = (world_pos[0] * 0.08 + world_pos[2] * 0.12).sin() * 0.5 + 0.5;

    // Determine main candy color based on noise
    let color_selector = color_noise1 + color_noise2 * 0.5;

    let mut base_color = if color_selector < 0.15 {
        // Hot pink / magenta candy
        mix_color([1.0, 0.2, 0.8], [1.0, 0.6, 0.9], stripe_pattern)
    } else if color_selector < 0.3 {
        // Electric blue candy
        mix_color([0.0, 0.8, 1.0], [0.4, 0.9, 1.0], stripe_pattern)
    } else if color_selector < 0.45 {
        // Lime green candy
        mix_color([0.5, 1.0, 0.2], [0.7, 1.0, 0.5], stripe_pattern)
    } else if color_selector < 0.6 {
        // Orange creamsicle
        mix_color([1.0, 0.5, 0.1], [1.0, 0.8, 0.3], stripe_pattern)
    } else if color_selector < 0.75 {
        // Purple grape candy
        mix_color([0.6, 0.2, 1.0], [0.8, 0.5, 1.0], stripe_pattern)
    } else if color_selector < 0.9 {
        // Yellow lemon candy
        mix_color([1.0, 1.0, 0.2], [1.0, 1.0, 0.6], stripe_pattern)
    } else {
        // Red cherry candy
        mix_color([1.0, 0.2, 0.3], [1.0, 0.5, 0.6], stripe_pattern)
    };

    // Add candy swirl effects
    let swirl_intensity = color_noise3 * swirl_pattern;
    if swirl_intensity > 0.7 {
        // White candy swirls
        base_color = mix_color(base_color, [1.0, 0.95, 1.0], 0.6);
    } else if swirl_intensity > 0.5 {
        // Pastel candy mixing
        let pastel_color = [
            base_color[0] * 0.7 + 0.3,
            base_color[1] * 0.7 + 0.3,
            base_color[2] * 0.7 + 0.3,
        ];
        base_color = mix_color(base_color, pastel_color, 0.4);
    }

    // Add rainbow zones
    let rainbow_noise = rock_voronoi3(
        [
            (world_pos[0] + 400.0) / 200.0,
            (world_pos[1] + 400.0) / 200.0,
            (world_pos[2] + 400.0) / 200.0,
        ],
        2.0,
        7,
    );
    if rainbow_noise > 0.8 {
        let hue = (world_pos[0] * 0.01 + world_pos[2] * 0.015 + rainbow_noise).fract();
        let rainbow_color = hsv_to_rgb([hue, 0.9, 1.0]);
        base_color = mix_color(base_color, rainbow_color, 0.5);
    }

    // Add sparkly highlights
    let sparkle_noise = rock_voronoi3(
        [
            (world_pos[0] + 500.0) / 25.0,
            (world_pos[1] + 500.0) / 25.0,
            (world_pos[2] + 500.0) / 25.0,
        ],
        7.0,
        2,
    );
    if sparkle_noise > 0.85 {
        // Bright white sparkles
        base_color = mix_color(base_color, [1.0, 1.0, 1.0], 0.8);
    } else if sparkle_noise > 0.75 {
        // Colored sparkles
        base_color = [
            base_color[0] * 1.3,
            base_color[1] * 1.3,
            base_color[2] * 1.3,
        ];
    }

    // Add surface detail variation
    let surface_variation = detail_noise - 0.5;
    base_color = [
        (base_color[0] + surface_variation * 0.2).clamp(0.0, 1.0),
        (base_color[1] + surface_variation * 0.2).clamp(0.0, 1.0),
        (base_color[2] + surface_variation * 0.2).clamp(0.0, 1.0),
    ];

    // Convert to packed RGBA (with full alpha)
    let r = (base_color[0] * 255.0) as u32 & 0xFF;
    let g = (base_color[1] * 255.0) as u32 & 0xFF;
    let b = (base_color[2] * 255.0) as u32 & 0xFF;
    let a = 255u32;

    (a << 24) | (b << 16) | (g << 8) | r
}
