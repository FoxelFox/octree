#import "perlin.wgsl"
#import "../data/context.wgsl"

struct Octree {
    data: atomic<u32>,
    childs: array<atomic<u32>, 8>
}

const INVALID_INDEX: u32 = 0xFFFFFFFFu; // Changed from 0u to avoid collision with root

// input
@group(1) @binding(0) var<uniform> context: Context;

// output - changed to store floating point SDF values
@group(0) @binding(0) var<storage, read_write> noise: array<f32>;
@group(0) @binding(1) var<storage, read_write> node_counter: atomic<u32>;
@group(0) @binding(2) var<storage, read_write> nodes: array<Octree>;

fn insert(index: u32, pos: vec3<u32>, depth: u32, data: u32) -> u32 {
    // Calculate the size of the current octree level
    let level_size =  context.grid_size >> depth;

    // Calculate which child this voxel belongs to at the current level
    let half_size = level_size >> 1u;

    // Calculate position within the current level
    let relative_pos = vec3<u32>(
        pos.x % level_size,
        pos.y % level_size,
        pos.z % level_size
    );

    // Determine the octant (0-7) based on position
    let octant_x = select(0u, 1u, relative_pos.x >= half_size);
    let octant_y = select(0u, 2u, relative_pos.y >= half_size);
    let octant_z = select(0u, 4u, relative_pos.z >= half_size);
    let octant = octant_x | octant_y | octant_z;

    // Try to atomically claim this child slot
    let old_value = atomicCompareExchangeWeak(
        &nodes[index].childs[octant],
        INVALID_INDEX,  // expected value
        0xFFFFFFFEu     // temporary placeholder (different from INVALID_INDEX)
    );

    var result_index: u32;

    if (old_value.exchanged) {
        // We successfully claimed the slot, now create the node
        let new_node_index = atomicAdd(&node_counter, 1u);

        // Initialize the new node
        atomicStore(&nodes[new_node_index].data, data);

        for (var i = 0u; i < 8u; i++) {
            atomicStore(&nodes[new_node_index].childs[i], INVALID_INDEX);
        }

        // Update parent's child pointer with the actual index
        atomicStore(&nodes[index].childs[octant], new_node_index);

        result_index = new_node_index;
    } else {
        // Another thread already created this node
        // Wait for the actual index to be written
        var actual_index = atomicLoad(&nodes[index].childs[octant]);
        var spin_count = 0u;
        while (actual_index == 0xFFFFFFFEu && spin_count < 1000u) {  // Wait with timeout
            actual_index = atomicLoad(&nodes[index].childs[octant]);
            spin_count = spin_count + 1u;
        }

        // Handle timeout case
        // TODO this should never happen
        if (actual_index == 0xFFFFFFFEu) {
            result_index = index; // Return parent on timeout
        } else {
            result_index = actual_index;
        }
    }

    return result_index;
}

// Generate continuous SDF values instead of binary 0/1
fn generate_sdf_noise(pos: vec3<u32>) -> f32 {
    // Generate deterministic noise that doesn't depend on time for TAA stability
    let noise_value = noise3(vec3<f32>(pos) / 20.0);

    // Convert noise from [0,1] range to SDF values
    // Values > 0.5 become negative (inside), values < 0.5 become positive (outside)
    let threshold = 0.5;
    let sdf_range = 2.0; // Maximum distance value

    return (threshold - noise_value) * sdf_range;
}

// Legacy function to maintain compatibility with octree generation
fn generate_noise(pos: vec3<u32>) -> u32 {
    let sdf_value = generate_sdf_noise(pos);
    return select(0u, 1u, sdf_value < 0.0);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // Initialize root node (only one thread should do this)
    if (id.x == 0u && id.y == 0u && id.z == 0u) {
        // Set node counter to 1 (root is at index 0)
        atomicStore(&node_counter, 1u);

        // Initialize root node
        atomicStore(&nodes[0].data, 0u);
        for (var i = 0u; i < 8u; i++) {
            atomicStore(&nodes[0].childs[i], INVALID_INDEX);
        }
    }

    // Ensure root initialization completes before other threads proceed
    workgroupBarrier();

    // Generate continuous SDF values
    let sdf_value = generate_sdf_noise(id);
    noise[to1D(id)] = sdf_value;

    // Generate binary value for octree (for compatibility)
    let zero_or_one = select(0u, 1u, sdf_value < 0.0);

    // Octree Generation
    if (zero_or_one == 1u) {
        var current_index = 0u; // Start from root


        for (var depth = 0u; depth < context.max_depth; depth++) {
        	// only store data in leafs othes have to be zeros
        	var data = select(0u, 1u, (depth + 1u) == context.max_depth);
            current_index = insert(current_index, id, depth, data);
        }
    }
}
