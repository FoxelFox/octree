#import "../data/context.wgsl"

struct VertexOutput {
  @builtin(position) position : vec4f,
}

struct Mesh {
	vertexCount: u32,
	vertices: array<vec4<f32>, 1024>, // 8 * 8 * 8 * 12 worst case of max verts
}

@group(0) @binding(0) var<storage, read> meshes: array<Mesh>;
@group(1) @binding(0) var <uniform> context: Context;

@vertex
fn vs_main(
      @builtin(vertex_index) vertexIndex: u32,
      @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
	var out: VertexOutput;
	
	// Skip if vertex index exceeds actual vertex count for this mesh
	if (vertexIndex >= meshes[instanceIndex].vertexCount) {
		out.position = vec4(0.0, 0.0, 0.0, 0.0); // Degenerate vertex
		return out;
	}
	
	out.position = context.perspective * context.view * meshes[instanceIndex].vertices[vertexIndex];
	return out;
}

@fragment
fn fm_main() -> @location(0) vec4f {
	return vec4(1,0,0,0.1);
}