#import "../data/context.wgsl"

struct DistanceNode {
    distance: f32,
    material_id: u32,
}

// Input: continuous SDF values from noise pipeline  
@group(0) @binding(0) var<storage, read> voxel_data: array<f32>;

// Output: distance field
@group(0) @binding(1) var<storage, read_write> distance_field: array<DistanceNode>;

@group(1) @binding(0) var<uniform> context: Context;

fn to1D(id: vec3<u32>) -> u32 {
    return id.z * context.grid_size * context.grid_size + id.y * context.grid_size + id.x;
}

// Get continuous SDF value at position
fn get_sdf_value(pos: vec3<i32>) -> f32 {
    let gs = i32(context.grid_size);
    if (pos.x < 0 || pos.y < 0 || pos.z < 0 || pos.x >= gs || pos.y >= gs || pos.z >= gs) {
        return 10.0; // Outside bounds = far from surface
    }
    let index = u32(pos.z * gs * gs + pos.y * gs + pos.x);
    return voxel_data[index];
}

// Legacy function for binary voxel access (for compatibility)
fn get_voxel(pos: vec3<i32>) -> u32 {
    let sdf_val = get_sdf_value(pos);
    return select(0u, 1u, sdf_val < 0.0);
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

// Smooth the SDF using trilinear interpolation of neighboring values
fn smooth_sdf_value(pos: vec3<u32>) -> f32 {
    let ipos = vec3<i32>(pos);
    let center_value = get_sdf_value(ipos);
    
    // Sample neighboring values for smoothing
    var sum = center_value * 8.0; // Center gets higher weight
    var weight = 8.0;
    
    // Add 6-connected neighbors
    let neighbors_6 = array<vec3<i32>, 6>(
        vec3<i32>(-1, 0, 0), vec3<i32>(1, 0, 0),
        vec3<i32>(0, -1, 0), vec3<i32>(0, 1, 0), 
        vec3<i32>(0, 0, -1), vec3<i32>(0, 0, 1)
    );
    
    for (var i = 0; i < 6; i++) {
        let neighbor_val = get_sdf_value(ipos + neighbors_6[i]);
        sum += neighbor_val * 4.0;
        weight += 4.0;
    }
    
    // Add 12 edge neighbors with lower weight
    let neighbors_12 = array<vec3<i32>, 12>(
        vec3<i32>(-1, -1, 0), vec3<i32>(-1, 1, 0), vec3<i32>(1, -1, 0), vec3<i32>(1, 1, 0),
        vec3<i32>(-1, 0, -1), vec3<i32>(-1, 0, 1), vec3<i32>(1, 0, -1), vec3<i32>(1, 0, 1),
        vec3<i32>(0, -1, -1), vec3<i32>(0, -1, 1), vec3<i32>(0, 1, -1), vec3<i32>(0, 1, 1)
    );
    
    for (var i = 0; i < 12; i++) {
        let neighbor_val = get_sdf_value(ipos + neighbors_12[i]);
        sum += neighbor_val * 2.0;
        weight += 2.0;
    }
    
    // Add 8 corner neighbors with lowest weight
    let neighbors_8 = array<vec3<i32>, 8>(
        vec3<i32>(-1, -1, -1), vec3<i32>(-1, -1, 1), vec3<i32>(-1, 1, -1), vec3<i32>(-1, 1, 1),
        vec3<i32>(1, -1, -1), vec3<i32>(1, -1, 1), vec3<i32>(1, 1, -1), vec3<i32>(1, 1, 1)
    );
    
    for (var i = 0; i < 8; i++) {
        let neighbor_val = get_sdf_value(ipos + neighbors_8[i]);
        sum += neighbor_val * 1.0;
        weight += 1.0;
    }
    
    return sum / weight;
}

@compute @workgroup_size(4, 4, 4) 
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= context.grid_size || id.y >= context.grid_size || id.z >= context.grid_size) {
        return;
    }
    
    let index = to1D(id);
    
    // Use the continuous SDF value directly with smoothing
    let raw_distance = get_sdf_value(vec3<i32>(id));
    let smoothed_distance = smooth_sdf_value(id);
    
    // Blend between raw and smoothed for optimal quality
    let distance = mix(raw_distance, smoothed_distance, 0.3);
    
    // Material ID: 0 = empty space, 1 = solid material
    let material_id = select(0u, 1u, distance < 0.0);
    
    distance_field[index] = DistanceNode(distance, material_id);
}