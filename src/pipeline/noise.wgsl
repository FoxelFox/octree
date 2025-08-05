#import "perlin.wgsl"
#import "../data/context.wgsl"

// input
@group(0) @binding(0) var <uniform> grid_size: u32;
@group(1) @binding(0) var <uniform> context: Context;

// output
@group(0) @binding(1) var<storage, read_write> noise: array<u32>;

fn to1D(id: vec3<u32>) -> u32 {
	return id.z * grid_size * grid_size + id.y * grid_size + id.x;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
	let random_value = noise3(vec3<f32>(id) / (2.0 * (sin(context.time/10) + 1.5)) - vec3<f32>(context.time, context.time, context.time) * 1 );
	var zero_or_one: u32;
	if (random_value > (sin(context.time / 10) * cos(context.time/10) * 0.5 + 0.5)) {
		zero_or_one = 1u;
	} else {
		zero_or_one = 0u;
	 }
    
	noise[to1D(id)] = zero_or_one;
}