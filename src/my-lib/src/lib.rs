extern crate wasm_bindgen;
mod noise;
mod simple;

use crate::simple::{generate_sin_color, generate_sin_noise};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
}

#[wasm_bindgen]
pub fn add(a: f32, b: f32) -> f32 {
    a + b
}

/// Generate noise data for a chunk at the given position
/// Returns Float32Array containing interleaved density and color data
/// Format: [density0, color0_as_f32, density1, color1_as_f32, ...]
/// Each pair represents one voxel (8 bytes total: f32 + u32 reinterpreted as f32)
#[wasm_bindgen]
pub fn noise_for_chunk(x: i32, y: i32, z: i32, size: u32) -> Box<[f32]> {
    let voxel_size = size + 1; // 257 for gridSize 256
    let chunk_offset = [x * size as i32, y * size as i32, z * size as i32];

    let total_voxels = (voxel_size * voxel_size * voxel_size) as usize;
    let mut result = Vec::with_capacity(total_voxels * 2); // density + color per voxel

    // Generate voxel data in the same order as WGSL: z * size² + y * size + x
    for vz in 0..voxel_size {
        for vy in 0..voxel_size {
            for vx in 0..voxel_size {
                let pos = [vx, vy, vz];
                // let density = generate_sdf_noise(pos, chunk_offset);
                // let color = generate_color(pos, chunk_offset, density);
                let density = generate_sin_noise(pos, chunk_offset);
                let color = generate_sin_color(pos, chunk_offset);

                // Store density as f32
                result.push(density);
                // Store color (u32) reinterpreted as f32 for buffer transfer
                result.push(f32::from_bits(color));
            }
        }
    }

    result.into_boxed_slice()
}
