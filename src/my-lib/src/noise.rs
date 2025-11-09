use wasm_bindgen::prelude::wasm_bindgen;

// Fixed world space size for one chunk
pub const SIZE: i32 = 256;

// Hand-crafted noise lookup table (256 values)
const NOISE_TABLE: [u8; 256] = [
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69,
    142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219,
    203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
    74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230,
    220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73, 209, 76,
    132, 187, 208, 89, 18, 169, 200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173,
    186, 3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206,
    59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163,
    70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232,
    178, 185, 112, 104, 218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162,
    241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204,
    176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141,
    128, 195, 78, 66, 215, 61, 156, 180,
];

fn noise_lookup(x: i32, y: i32, z: i32) -> f32 {
    let xi = (x & 255) as usize;
    let yi = (y & 255) as usize;
    let zi = (z & 255) as usize;

    let h1 = NOISE_TABLE[xi] as usize;
    let h2 = NOISE_TABLE[(h1 + yi) & 255] as usize;
    let h3 = NOISE_TABLE[(h2 + zi) & 255];

    // Normalize to -1.0 to 1.0
    (h3 as f32 / 127.5) - 1.0
}

fn sample_noise(x: f32, y: f32, z: f32) -> f32 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let z0 = z.floor() as i32;

    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let fz = z - z0 as f32;

    // Smoothstep interpolation
    let sx = fx * fx * (3.0 - 2.0 * fx);
    let sy = fy * fy * (3.0 - 2.0 * fy);
    let sz = fz * fz * (3.0 - 2.0 * fz);

    // Sample 8 corners of the cube
    let n000 = noise_lookup(x0, y0, z0);
    let n100 = noise_lookup(x0 + 1, y0, z0);
    let n010 = noise_lookup(x0, y0 + 1, z0);
    let n110 = noise_lookup(x0 + 1, y0 + 1, z0);
    let n001 = noise_lookup(x0, y0, z0 + 1);
    let n101 = noise_lookup(x0 + 1, y0, z0 + 1);
    let n011 = noise_lookup(x0, y0 + 1, z0 + 1);
    let n111 = noise_lookup(x0 + 1, y0 + 1, z0 + 1);

    // Trilinear interpolation
    let nx00 = n000 * (1.0 - sx) + n100 * sx;
    let nx10 = n010 * (1.0 - sx) + n110 * sx;
    let nx01 = n001 * (1.0 - sx) + n101 * sx;
    let nx11 = n011 * (1.0 - sx) + n111 * sx;

    let nxy0 = nx00 * (1.0 - sy) + nx10 * sy;
    let nxy1 = nx01 * (1.0 - sy) + nx11 * sy;

    nxy0 * (1.0 - sz) + nxy1 * sz
}

fn generate_sin_noise(pos: [f32; 3]) -> f32 {
    // Domain warping for organic distortion (only use X and Z for heightmap consistency)
    let warp_scale = 0.002;
    let warp_amount = 30.0;
    let warp_x = sample_noise(
        pos[0] * warp_scale,
        0.0, // Keep Y constant for heightmap terrain
        pos[2] * warp_scale,
    );
    let warp_z = sample_noise(
        (pos[0] + 73.2) * warp_scale,
        0.0, // Keep Y constant for heightmap terrain
        (pos[2] + 127.1) * warp_scale,
    );

    // Apply warping to position
    let warped_pos = [
        pos[0] + warp_x * warp_amount,
        pos[1],
        pos[2] + warp_z * warp_amount,
    ];

    // Fractal noise with 6 octaves - using much larger amplitudes for dramatic terrain
    let mut height = 0.0;
    let mut amplitude = 80.0; // Start with large amplitude for dramatic features
    let mut frequency = 0.003; // Start with large features
    let persistence = 0.5; // Each octave is 50% of previous
    let lacunarity = 2.0; // Each octave doubles frequency

    for i in 0..6 {
        // Sample noise only on horizontal plane for heightmap (2.5D terrain)
        let mut noise_val = sample_noise(
            warped_pos[0] * frequency,
            0.0, // Keep Y constant for proper heightmap
            warped_pos[2] * frequency,
        );

        // Use ridged noise for first 2 octaves (creates mountain ridges)
        if i < 2 {
            // Ridged noise: inverts and sharpens to create ridges
            noise_val = 1.0 - 2.0 * noise_val.abs(); // Maps to -1 to 1 range with sharp ridges
            noise_val = noise_val * noise_val * noise_val.signum(); // Sharpen the ridges (cubic)
        }
        // Use billowy noise for octave 2-3 (creates rolling hills)
        else if i == 2 || i == 3 {
            noise_val = noise_val.abs() * 2.0 - 1.0; // Billowy (puffy)
        }
        // Standard noise for fine detail (octaves 4, 5)

        height += noise_val * amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    // Add some valleys and erosion-like features
    let erosion_noise = sample_noise(
        pos[0] * 0.008,
        0.0,
        pos[2] * 0.008,
    );
    let erosion = erosion_noise * erosion_noise * 15.0; // Gentle valleys
    height -= erosion.abs();

    // Base terrain height - centered around y=32
    let surface_height = 64.0 + height;

    // SDF: distance from current Y to surface
    pos[1] - surface_height // negative below, positive above
}

fn generate_sin_color(pos: [f32; 3]) -> u32 {
    // Convert to packed RGBA (with full alpha)
    let r = ((pos[0] / 80.0).sin() * 255.0) as u32 & 0xFF;
    let g = ((pos[1] / 80.0).sin() * 255.0) as u32 & 0xFF;
    let b = ((pos[2] / 80.0).sin() * 255.0) as u32 & 0xFF;
    let a = 255u32;

    (a << 24) | (b << 16) | (g << 8) | r
}

/// Generate noise data for a chunk at the given position
/// Returns Float32Array containing interleaved density and color data
/// Format: [density0, color0_as_f32, density1, color1_as_f32, ...]
/// Each pair represents one voxel (8 bytes total: f32 + u32 reinterpreted as f32)
#[wasm_bindgen]
pub fn noise_for_chunk(x: i32, y: i32, z: i32, resolution: u32) -> Box<[f32]> {
    let voxel_size = resolution + 1; // 257 for gridSize 256
    let chunk_offset = [x * SIZE, y * SIZE, z * SIZE];

    let total_voxels = (voxel_size * voxel_size * voxel_size) as usize;
    let mut result = Vec::with_capacity(total_voxels * 2); // density + color per voxel

    // Generate voxel data in the same order as WGSL: z * size² + y * size + x
    for vz in 0..voxel_size {
        for vy in 0..voxel_size {
            for vx in 0..voxel_size {
                let pos = [vx, vy, vz];

                let world_pos = [
                    pos[0] as f32 + chunk_offset[0] as f32,
                    pos[1] as f32 + chunk_offset[1] as f32,
                    pos[2] as f32 + chunk_offset[2] as f32,
                ];

                let density = generate_sin_noise(world_pos);
                let color = generate_sin_color(world_pos);

                // Store density as f32
                result.push(density);
                // Store color (u32) reinterpreted as f32 for buffer transfer
                result.push(f32::from_bits(color));
            }
        }
    }

    result.into_boxed_slice()
}

pub fn only_noise_for_chunk(x: i32, y: i32, z: i32, resolution: u32, scale: f32) -> Vec<f32> {
    let voxel_size = resolution + 1; // 257 for gridSize 256
    let chunk_offset = [x * SIZE, y * SIZE, z * SIZE];

    let total_voxels = (voxel_size * voxel_size * voxel_size) as usize;
    let mut result = Vec::with_capacity(total_voxels);

    for vz in 0..voxel_size {
        for vy in 0..voxel_size {
            for vx in 0..voxel_size {
                let pos = [vx, vy, vz];
                let world_pos = [
                    pos[0] as f32 * scale + chunk_offset[0] as f32,
                    pos[1] as f32 * scale + chunk_offset[1] as f32,
                    pos[2] as f32 * scale + chunk_offset[2] as f32,
                ];
                let density = generate_sin_noise(world_pos);

                result.push(density);
            }
        }
    }

    result
}
