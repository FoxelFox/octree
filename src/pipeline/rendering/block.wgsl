#import "../../data/context.wgsl"

// Constants
const FOG_START = 0.0;
const FOG_END = 1750.0;

struct VertexOutput {
  @builtin(position) position : vec4f,
  @location(0) color: vec4<f32>,
  @location(1) world_pos: vec3<f32>,
  @location(2) normal: vec3<f32>,
}

@group(0) @binding(1) var<storage, read> vertices: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> normals: array<vec3<f32>>;
@group(0) @binding(3) var<storage, read> colors: array<u32>;
@group(1) @binding(0) var <uniform> context: Context;
@group(2) @binding(0) var space_background_texture: texture_2d<f32>;


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

	// Use vertexIndex directly - firstVertex in command handles offset
	let packedColor = colors[vertexIndex];
	let vertexColor = unpackColor(packedColor);
	out.color = vertexColor;

	out.world_pos = vertices[vertexIndex].xyz;

	out.normal = normals[vertexIndex];

	out.position = context.view_projection * vec4<f32>(vertices[vertexIndex]);
	return out;
}

// Convert ray direction to UV for texture sampling
fn ray_direction_to_uv(ray_dir: vec3<f32>) -> vec2<f32> {
    // Convert cartesian to spherical coordinates
    let theta = atan2(ray_dir.z, ray_dir.x); // Azimuth
    let phi = acos(ray_dir.y); // Elevation

    // Convert to UV coordinates
    let u = (theta + 3.14159265) / (2.0 * 3.14159265); // 0 to 1
    let v = phi / 3.14159265; // 0 to 1

    return vec2<f32>(u, v);
}

// Sample only nebula for fog (no stars)
fn sample_nebula_only(ray_dir: vec3<f32>) -> vec3<f32> {
    let uv = ray_direction_to_uv(ray_dir);
    let texture_size = textureDimensions(space_background_texture);
    let coord = vec2<i32>(uv * vec2<f32>(texture_size));
    return textureLoad(space_background_texture, coord, 0).rgb; // Only nebula
}

// Apply fog and environment to baked vertex colors
fn apply_fog_and_environment(diffuse_color: vec3<f32>, light_visibility: f32, world_normal: vec3<f32>, camera_pos: vec3<f32>, world_pos: vec3<f32>, distance: f32) -> vec3<f32> {
    // Add environment reflection scaled by light visibility (1 - shadow)
    let env_reflection = sample_nebula_only(world_normal) * 0.08 * light_visibility;
    let final_color = diffuse_color + diffuse_color * env_reflection;

    // Apply space fog - sample sky color in view direction
    let view_ray = normalize(world_pos - camera_pos);
    let fog_color = sample_nebula_only(view_ray);
    let fog_factor = clamp((FOG_END - distance) / (FOG_END - FOG_START), 0.0, 1.0);

    return mix(fog_color, final_color, fog_factor);
}

@fragment
fn fm_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Extract camera position from inverse view matrix
    let camera_pos = context.inverse_view[3].xyz;

    // Calculate distance from camera for fog
    let distance = length(camera_pos - in.world_pos);

    // Use baked vertex color directly
    let diffuse_color = in.color.rgb;
    let light_visibility = in.color.a;
    let world_normal = normalize(in.normal);

    // Apply fog and environment effects to baked colors
    let final_color = apply_fog_and_environment(
        diffuse_color,
        light_visibility,
        world_normal,
        camera_pos,
        in.world_pos,
        distance
    );

    return vec4<f32>(final_color, 1.0);
}
