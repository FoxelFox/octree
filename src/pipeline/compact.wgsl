#import "../data/context.wgsl"

// The new, highly optimized node structure
struct CompactNode {
    firstChildOrData: u32,
    childMask: u32,
}

// --- CONSTANTS ---
const LEAF_BIT: u32 = 0x80000000u;
const INVALID_INDEX: u32 = 0xFFFFFFFFu;
const FAT_NODE_STRIDE: u32 = 9u; // data (1) + childs (8) = 9

// --- BUFFERS ---
// We read the fat nodes as a raw u32 array to avoid atomic vs non-atomic struct mismatches.
@group(0) @binding(0) var<storage, read> fat_nodes: array<u32>;
@group(0) @binding(1) var<storage, read_write> compact_nodes: array<CompactNode>;
@group(0) @binding(2) var<storage, read_write> compact_node_counter: atomic<u32>;
@group(0) @binding(3) var<storage, read_write> leaf_node_counter: atomic<u32>;

var<workgroup> fat_node_stack: array<u32, 256>;
var<workgroup> compact_node_stack: array<u32, 256>;
var<workgroup> stack_ptr: i32;

fn process_node(fat_node_index: u32, compact_node_index: u32) {
    let base_offset = fat_node_index * FAT_NODE_STRIDE;
    let fat_node_data = fat_nodes[base_offset];

    var child_mask: u32 = 0u;
    var existing_child_count: u32 = 0u;

    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        if (fat_nodes[base_offset + 1u + i] != INVALID_INDEX) {
            child_mask = child_mask | (1u << i);
            existing_child_count = existing_child_count + 1u;
        }
    }

    if (child_mask == 0xFFu) {
        var first_child_data: u32 = 0u;
        var all_children_are_leaves = true;
        var all_data_is_the_same = true;

        for (var i: u32 = 0u; i < 8u; i = i + 1u) {
            let child_fat_node_index = fat_nodes[base_offset + 1u + i];
            let child_base_offset = child_fat_node_index * FAT_NODE_STRIDE;
            let child_fat_node_data = fat_nodes[child_base_offset];

            var is_leaf = true;
            for (var j: u32 = 0u; j < 8u; j = j + 1u) {
                if (fat_nodes[child_base_offset + 1u + j] != INVALID_INDEX) {
                    is_leaf = false;
                    break;
                }
            }

            if (!is_leaf) {
                all_children_are_leaves = false;
                break;
            }

            if (i == 0u) {
                first_child_data = child_fat_node_data;
            } else {
                if (child_fat_node_data != first_child_data) {
                    all_data_is_the_same = false;
                    break;
                }
            }
        }

        if (all_children_are_leaves && all_data_is_the_same) {
            compact_nodes[compact_node_index].firstChildOrData = first_child_data | LEAF_BIT;
            compact_nodes[compact_node_index].childMask = 0u;
            atomicAdd(&leaf_node_counter, 1u);
            return;
        }
    }

    if (existing_child_count == 0u) {
        compact_nodes[compact_node_index].firstChildOrData = fat_node_data | LEAF_BIT;
        compact_nodes[compact_node_index].childMask = 0u;
        atomicAdd(&leaf_node_counter, 1u);
    } else {
        let first_child_index = atomicAdd(&compact_node_counter, existing_child_count);
        compact_nodes[compact_node_index].firstChildOrData = first_child_index;
        compact_nodes[compact_node_index].childMask = child_mask;

        var processed_children: u32 = 0u;
        var i: i32 = 7;
        while(i >= 0) {
            let u_i = u32(i);
            if ((child_mask & (1u << u_i)) != 0u) {
                stack_ptr = stack_ptr + 1;
                fat_node_stack[stack_ptr] = fat_nodes[base_offset + 1u + u_i];
                compact_node_stack[stack_ptr] = first_child_index + existing_child_count - 1u - processed_children;
                processed_children = processed_children + 1u;
            }
            i = i - 1;
        }
    }
}

@compute @workgroup_size(1, 1, 1)
fn main() {
    atomicStore(&compact_node_counter, 1u);
    atomicStore(&leaf_node_counter, 0u);

    // Debug: increment leaf counter to prove shader runs
    atomicAdd(&leaf_node_counter, 42u);

    stack_ptr = 0;
    fat_node_stack[0] = 0u;
    compact_node_stack[0] = 0u;

    while (stack_ptr >= 0) {
        let fat_idx = fat_node_stack[stack_ptr];
        let compact_idx = compact_node_stack[stack_ptr];
        stack_ptr = stack_ptr - 1;
        process_node(fat_idx, compact_idx);
    }
}