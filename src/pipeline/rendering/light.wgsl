// Note: We define COMPRESSION locally since we don't use the full context import
const COMPRESSION = 8;

struct LightConfig {
    max_iterations: f32,
    light_attenuation: f32,
    shadow_softness: f32,
    skylight_intensity: f32,
    // Reserved for future use - using vec4 for proper 16-byte alignment
    _padding: vec4<f32>,
}

// Input: mesh density data
@group(0) @binding(0) var<storage, read> density: array<u32>;

// Output: light data (RG format - R=light intensity, G=shadow factor)
@group(0) @binding(1) var<storage, read_write> light_data: array<vec2<f32>>;

// Configuration
@group(0) @binding(2) var<uniform> config: LightConfig;

// Neighbor light buffers (6 directions: -X, +X, -Y, +Y, -Z, +Z)
@group(0) @binding(3) var<storage, read> neighbor_light_nx: array<vec2<f32>>; // -X neighbor
@group(0) @binding(4) var<storage, read> neighbor_light_px: array<vec2<f32>>; // +X neighbor
@group(0) @binding(5) var<storage, read> neighbor_light_ny: array<vec2<f32>>; // -Y neighbor
@group(0) @binding(6) var<storage, read> neighbor_light_py: array<vec2<f32>>; // +Y neighbor
@group(0) @binding(7) var<storage, read> neighbor_light_nz: array<vec2<f32>>; // -Z neighbor
@group(0) @binding(8) var<storage, read> neighbor_light_pz: array<vec2<f32>>; // +Z neighbor

// Context data (no longer using global context, using per-chunk data instead)
@group(1) @binding(0) var<uniform> chunk_world_pos: vec4<i32>;
@group(1) @binding(1) var<uniform> chunk_resolution: vec4<u32>; // vec4 for alignment, only .x is used

// Helper function to get the chunk's meshlet count per axis
fn chunk_meshlet_count() -> u32 {
    return chunk_resolution.x / u32(COMPRESSION);
}

// Convert 3D position to 1D index using chunk's actual meshlet count
fn to1D_chunk(id: vec3<u32>) -> u32 {
    let size = chunk_meshlet_count();
    return id.z * size * size + id.y * size + id.x;
}

// Check if coordinates are within bounds
fn isInBounds(pos: vec3<i32>) -> bool {
    let size = i32(chunk_meshlet_count());
    return pos.x >= 0 && pos.y >= 0 && pos.z >= 0 &&
           pos.x < size && pos.y < size && pos.z < size;
}

// Get density at position (returns 0 if out of bounds)
fn getDensity(pos: vec3<i32>) -> f32 {
    if (!isInBounds(pos)) {
        return 0.0;
    }
    let index = to1D_chunk(vec3<u32>(pos));
    return f32(density[index]) / (COMPRESSION * COMPRESSION * COMPRESSION);
}

// Get light data at position, including from neighbor chunks
fn getLightData(pos: vec3<i32>) -> vec2<f32> {
    let size = i32(chunk_meshlet_count());

    // Check if we need to sample from a neighbor chunk
    if (pos.x < 0) {
        // Sample from -X neighbor at their rightmost edge
        let neighbor_pos = vec3<i32>(size - 1, pos.y, pos.z);
        if (neighbor_pos.y >= 0 && neighbor_pos.y < size && neighbor_pos.z >= 0 && neighbor_pos.z < size) {
            let index = to1D_chunk(vec3<u32>(neighbor_pos));
            return neighbor_light_nx[index];
        }
        return vec2<f32>(0.0, 1.0);
    } else if (pos.x >= size) {
        // Sample from +X neighbor at their leftmost edge
        let neighbor_pos = vec3<i32>(0, pos.y, pos.z);
        if (neighbor_pos.y >= 0 && neighbor_pos.y < size && neighbor_pos.z >= 0 && neighbor_pos.z < size) {
            let index = to1D_chunk(vec3<u32>(neighbor_pos));
            return neighbor_light_px[index];
        }
        return vec2<f32>(0.0, 1.0);
    } else if (pos.y < 0) {
        // Sample from -Y neighbor
        let neighbor_pos = vec3<i32>(pos.x, size - 1, pos.z);
        if (neighbor_pos.x >= 0 && neighbor_pos.x < size && neighbor_pos.z >= 0 && neighbor_pos.z < size) {
            let index = to1D_chunk(vec3<u32>(neighbor_pos));
            return neighbor_light_ny[index];
        }
        return vec2<f32>(0.0, 1.0);
    } else if (pos.y >= size) {
        // Sample from +Y neighbor
        let neighbor_pos = vec3<i32>(pos.x, 0, pos.z);
        if (neighbor_pos.x >= 0 && neighbor_pos.x < size && neighbor_pos.z >= 0 && neighbor_pos.z < size) {
            let index = to1D_chunk(vec3<u32>(neighbor_pos));
            return neighbor_light_py[index];
        }
        return vec2<f32>(0.0, 1.0);
    } else if (pos.z < 0) {
        // Sample from -Z neighbor
        let neighbor_pos = vec3<i32>(pos.x, pos.y, size - 1);
        if (neighbor_pos.x >= 0 && neighbor_pos.x < size && neighbor_pos.y >= 0 && neighbor_pos.y < size) {
            let index = to1D_chunk(vec3<u32>(neighbor_pos));
            return neighbor_light_nz[index];
        }
        return vec2<f32>(0.0, 1.0);
    } else if (pos.z >= size) {
        // Sample from +Z neighbor
        let neighbor_pos = vec3<i32>(pos.x, pos.y, 0);
        if (neighbor_pos.x >= 0 && neighbor_pos.x < size && neighbor_pos.y >= 0 && neighbor_pos.y < size) {
            let index = to1D_chunk(vec3<u32>(neighbor_pos));
            return neighbor_light_pz[index];
        }
        return vec2<f32>(0.0, 1.0);
    }

    // Sample from current chunk
    let index = to1D_chunk(vec3<u32>(pos));
    return light_data[index];
}

// Calculate light propagation from neighboring cells
fn calculateLightPropagation(pos: vec3<i32>) -> vec2<f32> {
    let current_density = getDensity(pos);
    let is_solid = current_density > 0.5; // Threshold for solid voxels
    
    // If this cell is solid, it blocks light completely
    if (is_solid) {
        return vec2<f32>(0.0, 1.0); // No light, full shadow
    }
    
    // Check if we're completely enclosed (cave detection)
    var solid_neighbors = 0;
    let cave_check_dirs = array<vec3<i32>, 6>(
        vec3<i32>( 1,  0,  0), vec3<i32>(-1,  0,  0),
        vec3<i32>( 0,  1,  0), vec3<i32>( 0, -1,  0),
        vec3<i32>( 0,  0,  1), vec3<i32>( 0,  0, -1)
    );
    
    for (var i = 0; i < 6; i++) {
        let neighbor_pos = pos + cave_check_dirs[i];
        if (getDensity(neighbor_pos) > 0.3) {
            solid_neighbors++;
        }
    }
    
    // If surrounded by mostly solid blocks, force darkness (cave)
    if (solid_neighbors >= 5) {
        return vec2<f32>(0.0, 1.0); // Force cave darkness
    }
    
    let current_light = getLightData(pos);
    var max_light = current_light.x;
    var min_shadow = current_light.y;
    
    // Sample light from 6 neighboring directions
    let directions = array<vec3<i32>, 6>(
        vec3<i32>( 1,  0,  0), // +X
        vec3<i32>(-1,  0,  0), // -X
        vec3<i32>( 0,  1,  0), // +Y (up)
        vec3<i32>( 0, -1,  0), // -Y (down)
        vec3<i32>( 0,  0,  1), // +Z
        vec3<i32>( 0,  0, -1)  // -Z
    );
    
    // Weight factors for different directions (favor upward light propagation)
    let direction_weights = array<f32, 6>(
        0.8, 0.8, // Horizontal directions
        1.2,      // Upward (stronger skylight influence)
        0.6,      // Downward (less influence)
        0.8, 0.8  // Other horizontal directions
    );
    
    for (var i = 0; i < 6; i++) {
        let neighbor_pos = pos + directions[i];
        let neighbor_light = getLightData(neighbor_pos);
        let neighbor_density = getDensity(neighbor_pos);
        
        // Calculate light transmission through neighbor - be more aggressive about blocking
        let transmission = 1.0 - min(1.0, neighbor_density * 2.0); // More aggressive blocking
        var attenuated_light = neighbor_light.x * transmission * config.light_attenuation * direction_weights[i];
        
        // Don't propagate light if the neighbor is solid
        if (neighbor_density > 0.1) {
            attenuated_light = 0.0;
        }
        
        // Propagate light
        max_light = max(max_light, attenuated_light);
        
        // Calculate shadows - light is blocked by solid neighbors
        if (neighbor_density > 0.1) {
            let shadow_strength = neighbor_density * (1.0 - config.shadow_softness);
            min_shadow = min(min_shadow, shadow_strength);
        }
    }
    
    // Calculate world Y position for skylight
    // Each meshlet covers (256 / meshlet_count) world units
    let meshlet_count = chunk_meshlet_count();
    let meshlet_world_size = 256.0 / f32(meshlet_count);
    let world_y = f32(pos.y) * meshlet_world_size + f32(chunk_world_pos.y);

    // Check if there's a clear vertical path to the surface for skylight
    var has_sky_access = true;
    for (var check_y = pos.y + 1; check_y < i32(meshlet_count); check_y++) {
        let check_pos = vec3<i32>(pos.x, check_y, pos.z);
        if (getDensity(check_pos) > 0.3) {
            has_sky_access = false;
            break;
        }
    }

    // Only apply skylight if there's direct access to sky and above certain height
    // Skylight intensity based on world Y position (brighter near world top)
    if (has_sky_access && world_y > 0.0) {
        // More generous skylight - max at world Y > 256
        let skylight_factor = clamp(world_y / 256.0, 0.0, 1.0);
        let skylight = config.skylight_intensity * skylight_factor;
        max_light = max(max_light, skylight);
    }
    
    // Remove minimum ambient light - caves should be dark!
    
    return vec2<f32>(
        clamp(max_light, 0.0, 1.0),
        clamp(min_shadow, 0.0, 1.0)
    );
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let meshlet_count = chunk_meshlet_count();

    // Check bounds - early exit if beyond chunk's meshlet grid
    if (global_id.x >= meshlet_count || global_id.y >= meshlet_count || global_id.z >= meshlet_count) {
        return;
    }

    let pos = vec3<i32>(global_id);
    let index = to1D_chunk(global_id);

    // Calculate new light values
    let new_light = calculateLightPropagation(pos);

    // Store the updated light data
    light_data[index] = new_light;
}