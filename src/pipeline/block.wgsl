#import "../data/context.wgsl"

struct VertexOutput {
  @builtin(position) position : vec4f,
}

@group(0) @binding(0) var<storage> positions: array<vec4f>;
@group(1) @binding(0) var <uniform> context: Context;

@vertex
fn vs_main(
	@builtin(instance_index) id: u32,
	@location(0) vertex_position: vec4f
) -> VertexOutput {
	var out: VertexOutput;
	out.position = context.perspective * context.view * (vertex_position + positions[id]);
	return out;
}

@fragment
fn fm_main() -> @location(0) vec4f {
	return vec4(1,0,0,0.1);
}