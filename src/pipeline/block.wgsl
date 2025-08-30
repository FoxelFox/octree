#import "../data/context.wgsl"

struct VertexOutput {
  @builtin(position) position : vec4f,
  @location(0) color: vec3<f32>,
}

struct Mesh {
	vertexCount: u32,
	vertices: array<vec4<f32>, 2048>, // worst case is way larger than 2048
}

@group(0) @binding(0) var<storage, read> meshes: array<Mesh>;
@group(1) @binding(0) var <uniform> context: Context;

// Hash function for pseudo-random color generation
fn hash(x: u32) -> u32 {
    var h = x;
    h ^= h >> 16u;
    h *= 0x85ebca6bu;
    h ^= h >> 13u;
    h *= 0xc2b2ae35u;
    h ^= h >> 16u;
    return h;
}

// Generate random color from mesh index
fn randomColor(meshIndex: u32) -> vec3<f32> {
    let h1 = hash(meshIndex);
    let h2 = hash(meshIndex + 1u);
    let h3 = hash(meshIndex + 2u);
    
    return vec3<f32>(
        f32(h1 & 0xFFu) / 255.0,
        f32(h2 & 0xFFu) / 255.0,
        f32(h3 & 0xFFu) / 255.0
    );
}

@vertex
fn vs_main(
      @builtin(vertex_index) vertexIndex: u32,
      @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
	var out: VertexOutput;
	
	// Generate color once per vertex based on mesh index
	out.color = randomColor(instanceIndex);
	
	out.position = context.perspective * context.view * meshes[instanceIndex].vertices[vertexIndex];
	return out;
}

@fragment
fn fm_main(in: VertexOutput) -> @location(0) vec4f {
    return vec4(in.color, 1.0);
}