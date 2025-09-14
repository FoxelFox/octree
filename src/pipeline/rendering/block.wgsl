enable f16;

#import "../../data/context.wgsl"

struct VertexOutput {
  @builtin(position) position : vec4f,
  @location(0) color: vec3<f32>,
  @location(1) world_pos: vec3<f32>,
  @location(2) normal: vec3<f32>,
}

struct Mesh {
	vertexCount: u32,
	vertices: array<vec4<f16>, 1536>, // worst case is way larger than 2048
	normals: array<vec3<f16>, 1536>,
	colors: array<u32, 1536>, // Packed RGBA colors per vertex
}

@group(0) @binding(0) var<storage, read> meshes: array<Mesh>;
@group(1) @binding(0) var <uniform> context: Context;


// Unpack RGBA color from u32
fn unpackColor(packedColor: u32) -> vec4<f32> {
    let r = f32(packedColor & 0xFFu) / 255.0;
    let g = f32((packedColor >> 8u) & 0xFFu) / 255.0;
    let b = f32((packedColor >> 16u) & 0xFFu) / 255.0;
    let a = f32((packedColor >> 24u) & 0xFFu) / 255.0;
    return vec4<f32>(r, g, b, a);
}


@vertex
fn vs_main(
      @builtin(vertex_index) vertexIndex: u32,
      @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
	var out: VertexOutput;

	// Use per-vertex color from mesh data
	let packedColor = meshes[instanceIndex].colors[vertexIndex];
	let vertexColor = unpackColor(packedColor);
	out.color = vertexColor.rgb;

	// Get world position (convert from f16 to f32)
	let world_pos = vec3<f32>(meshes[instanceIndex].vertices[vertexIndex].xyz);
	out.world_pos = world_pos;

	// Use stored normal (convert from f16 to f32)
	out.normal = vec3<f32>(meshes[instanceIndex].normals[vertexIndex]);

	out.position = context.perspective * context.view * vec4<f32>(meshes[instanceIndex].vertices[vertexIndex]);
	return out;
}

// G-buffer output structure
struct GBufferOutput {
    @location(0) position: vec4<f32>,  // xyz = world position, w = depth
    @location(1) normal: vec4<f32>,    // xyz = world normal, w = unused
    @location(2) diffuse: vec4<f32>,   // xyz = diffuse color, w = unused
}

@fragment
fn fm_main(in: VertexOutput) -> GBufferOutput {
    var output: GBufferOutput;

    // Extract camera position from inverse view matrix
    let camera_pos = context.inverse_view[3].xyz;

    // Calculate distance from camera for depth
    let distance = length(camera_pos - in.world_pos);

    // Output world position and depth
    output.position = vec4<f32>(in.world_pos, distance);

    // Output world normal
    output.normal = vec4<f32>(normalize(in.normal), 1.0);

    // Output diffuse color
    output.diffuse = vec4<f32>(in.color, 1.0);

    return output;
}
