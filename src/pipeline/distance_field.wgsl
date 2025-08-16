#import "../data/context.wgsl"

// Input: voxel data from noise pipeline
@group(0) @binding(0) var<storage, read> voxel_data: array<u32>;

// Output: distance field
@group(0) @binding(1) var<storage, read_write> distance_field: array<DistanceNode>;

@group(1) @binding(0) var<uniform> context: Context;

struct DistanceNode {
    distance: f32,
    material_id: u32,
}

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

// Simple distance field generation using 3x3x3 neighborhood
fn calculate_distance(pos: vec3<u32>) -> f32 {
    let center_voxel = get_voxel(vec3<i32>(pos));
    let ipos = vec3<i32>(pos);
    
    if (center_voxel == 1u) {
        // We're inside a voxel - find distance to nearest empty space
        var min_dist = 999.0;
        var found_empty = false;
        
        // Check 3x3x3 neighborhood
        for (var dx = -1; dx <= 1; dx++) {
            for (var dy = -1; dy <= 1; dy++) {
                for (var dz = -1; dz <= 1; dz++) {
                    let neighbor_pos = ipos + vec3<i32>(dx, dy, dz);
                    let neighbor_voxel = get_voxel(neighbor_pos);
                    
                    if (neighbor_voxel == 0u) {
                        found_empty = true;
                        let dist = length(vec3<f32>(f32(dx), f32(dy), f32(dz)));
                        min_dist = min(min_dist, dist);
                    }
                }
            }
        }
        
        if (found_empty) {
            return -min_dist; // Negative = inside
        } else {
            return -1.0; // Deep inside
        }
    } else {
        // We're outside a voxel - find distance to nearest solid space
        var min_dist = 999.0;
        var found_solid = false;
        
        // Check 3x3x3 neighborhood
        for (var dx = -1; dx <= 1; dx++) {
            for (var dy = -1; dy <= 1; dy++) {
                for (var dz = -1; dz <= 1; dz++) {
                    let neighbor_pos = ipos + vec3<i32>(dx, dy, dz);
                    let neighbor_voxel = get_voxel(neighbor_pos);
                    
                    if (neighbor_voxel == 1u) {
                        found_solid = true;
                        let dist = length(vec3<f32>(f32(dx), f32(dy), f32(dz)));
                        min_dist = min(min_dist, dist);
                    }
                }
            }
        }
        
        if (found_solid) {
            return min_dist; // Positive = outside
        } else {
            return 1.0; // Deep outside
        }
    }
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= context.grid_size || id.y >= context.grid_size || id.z >= context.grid_size) {
        return;
    }
    
    let index = to1D(id);
    let distance = calculate_distance(id);
    let material_id = get_voxel(vec3<i32>(id));
    
    distance_field[index] = DistanceNode(distance, material_id);
}