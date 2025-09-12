#import "../data/context.wgsl"

struct EditParams {
    position: vec3<f32>,
    radius: f32,
    operation: f32, // 0.0 = remove, 1.0 = add
    // padding to 32 bytes
}

// Voxel data structure containing density and color
struct VoxelData {
    density: f32,
    color: u32, // Packed RGBA color
}

// Input/Output
@group(0) @binding(0) var<storage, read_write> voxels: array<VoxelData>;
@group(0) @binding(1) var<uniform> edit_params: EditParams;

// Context
@group(1) @binding(0) var<uniform> context: Context;

// Convert world position to voxel grid coordinates
fn worldToVoxel(world_pos: vec3<f32>) -> vec3<i32> {
    // Assuming the voxel grid spans from 0 to gridSize in world coordinates
    // Adjust this based on your world-to-voxel coordinate mapping
    let voxel_pos = world_pos; // Simple 1:1 mapping, adjust as needed
    return vec3<i32>(voxel_pos);
}

// Calculate distance between two points
fn distance(a: vec3<f32>, b: vec3<f32>) -> f32 {
    let diff = a - b;
    return sqrt(dot(diff, diff));
}

// Smooth minimum function for blending operations
fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

// Smooth maximum function for blending operations
fn smax(a: f32, b: f32, k: f32) -> f32 {
    let h = clamp(0.5 - 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) + k * h * (1.0 - h);
}

// Neon pink gummy color as packed u32 (ABGR format: A|B|G|R)
const NEON_PINK_COLOR: u32 = 0xFF3185FF; // RGB(255,0,153) in ABGR format

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // Check bounds
    if (id.x >= context.grid_size || id.y >= context.grid_size || id.z >= context.grid_size) {
        return;
    }

    let voxel_index = to1D(id);
    let current_voxel = voxels[voxel_index];

    // Convert voxel coordinates to world position
    let voxel_world_pos = vec3<f32>(id);

    // Calculate distance from edit center
    let dist_to_center = distance(voxel_world_pos, edit_params.position);

    // Only modify voxels within the edit radius
    if (dist_to_center > edit_params.radius) {
        return;
    }

    // Create a sphere influence that falls off from center to edge
    let sphere_influence = 1.0 - (dist_to_center / edit_params.radius);
    let clamped_influence = clamp(sphere_influence, 0.0, 1.0);

    // Apply the edit operation to both density and color
    var new_voxel: VoxelData;

    if (edit_params.operation > 0.5) {
        // Add operation: push values negative (inside) where sphere influence is strong
        let add_strength = clamped_influence * 2.0; // Match noise range
        new_voxel.density = current_voxel.density - add_strength;

        // Completely replace with neon pink color
        new_voxel.color = NEON_PINK_COLOR;

    } else {
        // Remove operation: push values positive (outside) where sphere influence is strong
        let remove_strength = clamped_influence * 2.0; // Match noise range
        new_voxel.density = current_voxel.density + remove_strength;

        // For remove operations, keep the original color (voxel will be removed anyway)
        new_voxel.color = current_voxel.color;
    }

    // Write back the modified voxel data
    voxels[voxel_index] = new_voxel;
}
