#import "../data/context.wgsl"

struct VertexOutput {
  @builtin(position) position : vec4f,
  @location(0) color: vec3<f32>,
  @location(1) world_pos: vec3<f32>,
  @location(2) normal: vec3<f32>,
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

fn calculateNormal(v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>) -> vec3<f32> {
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    return normalize(cross(edge1, edge2));
}

@vertex
fn vs_main(
      @builtin(vertex_index) vertexIndex: u32,
      @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
	var out: VertexOutput;
	
	// Generate color once per vertex based on mesh index
	out.color = randomColor(instanceIndex);
	
	// Get world position
	let world_pos = meshes[instanceIndex].vertices[vertexIndex].xyz;
	out.world_pos = world_pos;
	
	// Calculate normal from triangle (assuming triangulated mesh)
	let triangleIndex = vertexIndex / 3u;
	let baseVertex = triangleIndex * 3u;
	
	if (baseVertex + 2u < meshes[instanceIndex].vertexCount) {
		let v0 = meshes[instanceIndex].vertices[baseVertex].xyz;
		let v1 = meshes[instanceIndex].vertices[baseVertex + 1u].xyz;
		let v2 = meshes[instanceIndex].vertices[baseVertex + 2u].xyz;
		out.normal = calculateNormal(v0, v1, v2);
	} else {
		out.normal = vec3<f32>(0.0, 1.0, 0.0); // Default up normal
	}
	
	out.position = context.perspective * context.view * meshes[instanceIndex].vertices[vertexIndex];
	return out;
}

@fragment
fn fm_main(in: VertexOutput) -> @location(0) vec4f {
    // Extract camera position from inverse view matrix
    let camera_pos = context.inverse_view[3].xyz;
    
    // Calculate distance from camera
    let distance = length(camera_pos - in.world_pos);
    
    // Light falloff parameters
    let max_light_distance = 256.0; // Maximum distance for light effect
    let falloff_start = 20.0; // Distance where falloff begins
    
    // Calculate distance attenuation
    var light_intensity = 1.0;
    if (distance > falloff_start) {
        let falloff_range = max_light_distance - falloff_start;
        let falloff_factor = max(0.0, (max_light_distance - distance) / falloff_range);
        light_intensity = falloff_factor * falloff_factor; // Quadratic falloff
    }
    
    // Light direction from fragment to camera (light at camera position)
    let light_dir = normalize(camera_pos - in.world_pos);
    
    // Simple diffuse lighting
    let diffuse = max(dot(in.normal, light_dir), 0.0);
    
    // Ambient lighting to prevent completely black surfaces
    let ambient = 0.3;
    
    // Apply distance attenuation to diffuse lighting only
    let lighting = ambient + diffuse * light_intensity;
    
    // Apply lighting to color
    let lit_color = in.color * lighting;
    
    // Fog parameters
    let fog_color = vec3<f32>(0.125, 0.125, 0.125); // Gray background color
    let fog_start = 50.0;
    let fog_end = 600.0;
    
    // Calculate fog factor (0 = full fog, 1 = no fog)
    let fog_factor = clamp((fog_end - distance) / (fog_end - fog_start), 0.0, 1.0);
    
    // Mix lit color with fog color
    let final_color = mix(fog_color, lit_color.rgb, fog_factor);
    
    return vec4(final_color, 1.0);
}