#import "../data/context.wgsl"

struct DistanceNode {
    distance: f32,
    material_id: u32,
}

// Input: voxel data from noise pipeline
@group(0) @binding(0) var<storage, read> voxel_data: array<u32>;

// Output: distance field
@group(0) @binding(1) var<storage, read_write> distance_field: array<DistanceNode>;

@group(1) @binding(0) var<uniform> context: Context;

fn to1D(id: vec3<u32>) -> u32 {
    return id.z * context.grid_size * context.grid_size + id.y * context.grid_size + id.x;
}

fn get_voxel(pos: vec3<i32>) -> u32 {
    let gs = i32(context.grid_size);
    if (pos.x < 0 || pos.y < 0 || pos.z < 0 || pos.x >= gs || pos.y >= gs || pos.z >= gs) {
        return 0u; // Outside bounds = empty
    }
    let index = u32(pos.z * gs * gs + pos.y * gs + pos.x);
    return voxel_data[index];
}

// High-precision distance field generation using larger neighborhood
fn calculate_distance(pos: vec3<u32>) -> f32 {
    let center_voxel = get_voxel(vec3<i32>(pos));
    let ipos = vec3<i32>(pos);
    
    if (center_voxel == 1u) {
        // We're inside a voxel - find distance to nearest empty space
        var min_dist = 999.0;
        
        // Check much larger neighborhood for better distance estimation
        for (var dx = -6; dx <= 6; dx++) {
            for (var dy = -6; dy <= 6; dy++) {
                for (var dz = -6; dz <= 6; dz++) {
                    let neighbor_pos = ipos + vec3<i32>(dx, dy, dz);
                    let neighbor_voxel = get_voxel(neighbor_pos);
                    
                    if (neighbor_voxel == 0u) {
                        let dist = length(vec3<f32>(f32(dx), f32(dy), f32(dz)));
                        min_dist = min(min_dist, dist);
                    }
                }
            }
        }
        
        if (min_dist < 999.0) {
            return -min_dist; // Negative = inside
        } else {
            return -6.0; // Deep inside
        }
    } else {
        // We're outside a voxel - find distance to nearest solid space
        var min_dist = 999.0;
        
        // Check much larger neighborhood for better distance estimation
        for (var dx = -6; dx <= 6; dx++) {
            for (var dy = -6; dy <= 6; dy++) {
                for (var dz = -6; dz <= 6; dz++) {
                    let neighbor_pos = ipos + vec3<i32>(dx, dy, dz);
                    let neighbor_voxel = get_voxel(neighbor_pos);
                    
                    if (neighbor_voxel == 1u) {
                        let dist = length(vec3<f32>(f32(dx), f32(dy), f32(dz)));
                        min_dist = min(min_dist, dist);
                    }
                }
            }
        }
        
        if (min_dist < 999.0) {
            return min_dist; // Positive = outside
        } else {
            return 6.0; // Deep outside
        }
    }
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= context.grid_size || id.y >= context.grid_size || id.z >= context.grid_size) {
        return;
    }
    
    let index = to1D(id);
    let voxel_value = get_voxel(vec3<i32>(id));
    
    // High-precision distance field: calculate actual distance to nearest surface
    var distance: f32;
    if (voxel_value == 1u) {
        // Inside: find distance to nearest empty voxel with larger neighborhood
        var min_dist = 6.0;
        for (var dx = -6; dx <= 6; dx++) {
            for (var dy = -6; dy <= 6; dy++) {
                for (var dz = -6; dz <= 6; dz++) {
                    let neighbor_pos = vec3<i32>(id) + vec3<i32>(dx, dy, dz);
                    if (get_voxel(neighbor_pos) == 0u) {
                        let dist = length(vec3<f32>(f32(dx), f32(dy), f32(dz)));
                        min_dist = min(min_dist, dist);
                    }
                }
            }
        }
        distance = -min_dist; // Negative for inside
    } else {
        // Outside: find distance to nearest solid voxel with larger neighborhood
        var min_dist = 6.0;
        for (var dx = -6; dx <= 6; dx++) {
            for (var dy = -6; dy <= 6; dy++) {
                for (var dz = -6; dz <= 6; dz++) {
                    let neighbor_pos = vec3<i32>(id) + vec3<i32>(dx, dy, dz);
                    if (get_voxel(neighbor_pos) == 1u) {
                        let dist = length(vec3<f32>(f32(dx), f32(dy), f32(dz)));
                        min_dist = min(min_dist, dist);
                    }
                }
            }
        }
        distance = min_dist; // Positive for outside
    }
    
    // Material ID: 0 = empty space, 1 = solid material
    let material_id = select(0u, 1u, voxel_value == 1u);
    
    distance_field[index] = DistanceNode(distance, material_id);
}