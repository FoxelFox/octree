#import "perlin.wgsl"
#import "../data/context.wgsl"

struct Octree {
  data: u32,
  parent: u32,
  childs: array<u32, 8>
}

// input
@group(0) @binding(0) var <uniform> grid_size: u32;
@group(1) @binding(0) var <uniform> context: Context;

// output
@group(0) @binding(1) var<storage, read_write> octree: array<Octree>;
@group(0) @binding(2) var<storage, read_write> pointer: atomic<u32>;

fn to1D(id: vec3<u32>) -> u32 {
  return id.z * grid_size * grid_size + id.y * grid_size + id.x;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let fuckyou=grid_size;

  let random_value = noise3(vec3<f32>(id) / (25.0 * (sin(context.time/10) + 1.5)) - vec3<f32>(context.time, context.time, context.time) * 1 );
  var zero_or_one: u32;
  if (random_value > (sin(context.time / 10) * cos(context.time/10) * 0.5 + 0.5)) {
    zero_or_one = 1u;

    var node: Octree;
    node.data = zero_or_one;
    let local_pointer = atomicAdd(&pointer, 1);
    octree[local_pointer] = node;      
  } else {
    zero_or_one = 0u;
  }
    

}