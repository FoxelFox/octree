#import "../../data/context.wgsl"

struct Command {
	vertexCount: u32,
	instanceCount: u32,
	firstVertex: u32,
	firstInstance: u32,
}

// Input: vertex data
@group(0) @binding(0) var<storage, read> vertices: array<vec4<f32>>;

// Material colors (read-only, original colors from mesh generation)
@group(0) @binding(1) var<storage, read> materialColors: array<u32>;

// Vertex normals
@group(0) @binding(2) var<storage, read> normals: array<vec3<f32>>;

// Lit colors (write-only, output with lighting applied)
@group(0) @binding(3) var<storage, read_write> colors: array<u32>;

// Vertex counts per meshlet
@group(0) @binding(4) var<storage, read> vertexCounts: array<u32>;

// Indirect draw commands (provides firstVertex offsets)
@group(0) @binding(5) var<storage, read> commands: array<Command>;

// Light data (chunk + 6 neighbors packed sequentially)
@group(1) @binding(0) var<storage, read> light_data: array<vec2<f32>>;

// Context
@group(2) @binding(0) var<uniform> context: Context;
@group(2) @binding(1) var<uniform> chunk_world_pos: vec3<i32>;

// Convert world position to chunk-local compressed grid coordinates
fn worldToCompressedGrid(world_pos: vec3<f32>) -> vec3<f32> {
    let chunk_local_pos = world_pos - vec3<f32>(chunk_world_pos);
    return chunk_local_pos / COMPRESSION;
}

const SEGMENT_CURRENT: u32 = 0u;
const SEGMENT_NX: u32 = 1u;
const SEGMENT_PX: u32 = 2u;
const SEGMENT_NY: u32 = 3u;
const SEGMENT_PY: u32 = 4u;
const SEGMENT_NZ: u32 = 5u;
const SEGMENT_PZ: u32 = 6u;
const NO_LIGHT = vec2<f32>(0.0, 1.0);

fn lightGridSize() -> u32 {
    return context.grid_size / COMPRESSION;
}

fn lightSegmentStride() -> u32 {
    let size = lightGridSize();
    return size * size * size;
}

fn clampScalar(value: i32, min_value: i32, max_value: i32) -> i32 {
    return max(min(value, max_value), min_value);
}

fn clampVec3(value: vec3<i32>, min_value: i32, max_value: i32) -> vec3<i32> {
    return vec3<i32>(
        clampScalar(value.x, min_value, max_value),
        clampScalar(value.y, min_value, max_value),
        clampScalar(value.z, min_value, max_value),
    );
}

fn inRange(value: i32, size: i32) -> bool {
    return value >= 0 && value < size;
}

fn sampleLightSegment(segment: u32, coords: vec3<i32>) -> vec2<f32> {
    let size_i = i32(lightGridSize());
    let max_index = size_i - 1;
    let clamped = clampVec3(coords, 0, max_index);
    let stride = lightSegmentStride();
    let size_sq = size_i * size_i;
    let index_i = clamped.z * size_sq + clamped.y * size_i + clamped.x;
    return light_data[segment * stride + u32(index_i)];
}

// Get light data at specific grid coordinates, sampling neighbor chunks when needed
fn getLightDataAtGrid(grid_pos: vec3<i32>) -> vec2<f32> {
    let size = i32(lightGridSize());

    if (grid_pos.x < 0) {
        if (inRange(grid_pos.y, size) && inRange(grid_pos.z, size)) {
            let neighbor_pos = vec3<i32>(size - 1, grid_pos.y, grid_pos.z);
            return sampleLightSegment(SEGMENT_NX, neighbor_pos);
        }
        return NO_LIGHT;
    } else if (grid_pos.x >= size) {
        if (inRange(grid_pos.y, size) && inRange(grid_pos.z, size)) {
            let neighbor_pos = vec3<i32>(0, grid_pos.y, grid_pos.z);
            return sampleLightSegment(SEGMENT_PX, neighbor_pos);
        }
        return NO_LIGHT;
    }

    if (grid_pos.y < 0) {
        if (inRange(grid_pos.x, size) && inRange(grid_pos.z, size)) {
            let neighbor_pos = vec3<i32>(grid_pos.x, size - 1, grid_pos.z);
            return sampleLightSegment(SEGMENT_NY, neighbor_pos);
        }
        return NO_LIGHT;
    } else if (grid_pos.y >= size) {
        if (inRange(grid_pos.x, size) && inRange(grid_pos.z, size)) {
            let neighbor_pos = vec3<i32>(grid_pos.x, 0, grid_pos.z);
            return sampleLightSegment(SEGMENT_PY, neighbor_pos);
        }
        return NO_LIGHT;
    }

    if (grid_pos.z < 0) {
        if (inRange(grid_pos.x, size) && inRange(grid_pos.y, size)) {
            let neighbor_pos = vec3<i32>(grid_pos.x, grid_pos.y, size - 1);
            return sampleLightSegment(SEGMENT_NZ, neighbor_pos);
        }
        return NO_LIGHT;
    } else if (grid_pos.z >= size) {
        if (inRange(grid_pos.x, size) && inRange(grid_pos.y, size)) {
            let neighbor_pos = vec3<i32>(grid_pos.x, grid_pos.y, 0);
            return sampleLightSegment(SEGMENT_PZ, neighbor_pos);
        }
        return NO_LIGHT;
    }

    return sampleLightSegment(SEGMENT_CURRENT, grid_pos);
}

// Sample light data with trilinear interpolation
fn sampleLightData(world_pos: vec3<f32>) -> vec2<f32> {
    let grid_pos = worldToCompressedGrid(world_pos);
    let base_pos = floor(grid_pos);
    let fract_pos = grid_pos - base_pos;

    let p000 = getLightDataAtGrid(vec3<i32>(base_pos));
    let p001 = getLightDataAtGrid(vec3<i32>(base_pos + vec3<f32>(0.0, 0.0, 1.0)));
    let p010 = getLightDataAtGrid(vec3<i32>(base_pos + vec3<f32>(0.0, 1.0, 0.0)));
    let p011 = getLightDataAtGrid(vec3<i32>(base_pos + vec3<f32>(0.0, 1.0, 1.0)));
    let p100 = getLightDataAtGrid(vec3<i32>(base_pos + vec3<f32>(1.0, 0.0, 0.0)));
    let p101 = getLightDataAtGrid(vec3<i32>(base_pos + vec3<f32>(1.0, 0.0, 1.0)));
    let p110 = getLightDataAtGrid(vec3<i32>(base_pos + vec3<f32>(1.0, 1.0, 0.0)));
    let p111 = getLightDataAtGrid(vec3<i32>(base_pos + vec3<f32>(1.0, 1.0, 1.0)));

    let c00 = mix(p000, p100, fract_pos.x);
    let c01 = mix(p001, p101, fract_pos.x);
    let c10 = mix(p010, p110, fract_pos.x);
    let c11 = mix(p011, p111, fract_pos.x);

    let c0 = mix(c00, c10, fract_pos.y);
    let c1 = mix(c01, c11, fract_pos.y);

    return mix(c0, c1, fract_pos.z);
}

// Sample directional light from the floodfill data
fn sampleDirectionalLight(world_pos: vec3<f32>, direction: vec3<f32>, step_size: f32) -> f32 {
    let sample_pos = world_pos + direction * step_size;
    let light_info = sampleLightData(sample_pos);
    return light_info.x;
}

// Unpack RGBA color from u32
fn unpackColor(packedColor: u32) -> vec4<f32> {
    let r = f32(packedColor & 0xFFu) / 255.0;
    let g = f32((packedColor >> 8u) & 0xFFu) / 255.0;
    let b = f32((packedColor >> 16u) & 0xFFu) / 255.0;
    let a = f32((packedColor >> 24u) & 0xFFu) / 255.0;
    return vec4<f32>(r, g, b, a);
}

// Pack RGBA color to u32
fn packColor(color: vec4<f32>) -> u32 {
    let r = u32(clamp(color.r, 0.0, 1.0) * 255.0) & 0xFFu;
    let g = u32(clamp(color.g, 0.0, 1.0) * 255.0) & 0xFFu;
    let b = u32(clamp(color.b, 0.0, 1.0) * 255.0) & 0xFFu;
    let a = u32(clamp(color.a, 0.0, 1.0) * 255.0) & 0xFFu;
    return (a << 24u) | (b << 16u) | (g << 8u) | r;
}

const MAX_WORKGROUPS_PER_DIM: u32 = 65535u;
const WORKGROUP_SIZE: u32 = 64u;
const VERTICES_PER_MESHLET: u32 = 1536u;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    // Flatten 2D workgroup dispatch into linear index
    let linear_workgroup = workgroup_id.y * MAX_WORKGROUPS_PER_DIM + workgroup_id.x;
    let linear_index = linear_workgroup * WORKGROUP_SIZE + local_index;

    // Compute which meshlet this invocation belongs to
    let meshlet_index = linear_index / VERTICES_PER_MESHLET;
    let vertex_in_meshlet = linear_index % VERTICES_PER_MESHLET;

    // Bounds check: ensure we don't exceed meshlet count
    if (meshlet_index >= arrayLength(&vertexCounts) || meshlet_index >= arrayLength(&commands)) {
        return;
    }

    let declared_vertex_count = vertexCounts[meshlet_index];
    let command = commands[meshlet_index];
    let vertex_count = min(declared_vertex_count, command.vertexCount);

    if (vertex_count == 0u) {
        return;
    }

    // Early exit if this vertex slot is beyond the meshlet's actual vertex count
    if (vertex_in_meshlet >= vertex_count) {
        return;
    }

    // Calculate actual vertex index in the vertex buffer
    let vertex_index = command.firstVertex + vertex_in_meshlet;
    if (vertex_index >= arrayLength(&vertices) || vertex_index >= arrayLength(&normals) || vertex_index >= arrayLength(&materialColors)) {
        return;
    }

    // Get vertex position in world space
    let vertex = vertices[vertex_index];
    let world_pos = vec3<f32>(
        f32(vertex.x),
        f32(vertex.y),
        f32(vertex.z)
    );

    // Get vertex normal
    let world_normal = normalize(normals[vertex_index]);

    // Sample voxel-based lighting data at vertex position
    let light_info = sampleLightData(world_pos);
    let base_light_intensity = light_info.x;
    let shadow_factor = clamp(light_info.y, 0.0, 1.0);

    // Directional lighting similar to deferred shader
    let sky_dir = vec3<f32>(0.0, 1.0, 0.0);
    let sky_light = sampleDirectionalLight(world_pos, sky_dir, 4.0);
    let sky_contribution = max(dot(-world_normal, sky_dir), 0.0) * sky_light * 0.8;

    let directions = array<vec3<f32>, 6>(
        vec3<f32>( 1.0,  0.0,  0.0),
        vec3<f32>(-1.0,  0.0,  0.0),
        vec3<f32>( 0.0,  1.0,  0.0),
        vec3<f32>( 0.0, -1.0,  0.0),
        vec3<f32>( 0.0,  0.0,  1.0),
        vec3<f32>( 0.0,  0.0, -1.0)
    );

    var directional_light = 0.0;
    for (var i = 0; i < 6; i++) {
        let dir_light = sampleDirectionalLight(world_pos, directions[i], 2.0);
        let normal_factor = max(dot(-world_normal, directions[i]), 0.0);
        directional_light += dir_light * normal_factor;
    }
    directional_light *= 0.15;

    let total_lighting = base_light_intensity * 0.6 + sky_contribution + directional_light;
    let exposure_adjustment = 1.0 / (1.0 + total_lighting * 0.3);
    let contrast_preserved_lighting = total_lighting * exposure_adjustment;
    let clamped_lighting = clamp(contrast_preserved_lighting, 0.0, 1.1);

    // Read material color (original color from mesh generation)
    let material_color = unpackColor(materialColors[vertex_index]);

    // Apply lighting and carry light visibility for deferred environment
    let light_visibility = clamp(1.0 - shadow_factor, 0.0, 1.0);
    let lit_color = material_color.rgb * clamped_lighting;
    let final_color = clamp(lit_color, vec3<f32>(0.0), vec3<f32>(1.0));

    // Store light visibility (1 - shadow) in alpha for the deferred pass
    colors[vertex_index] = packColor(vec4<f32>(final_color, light_visibility));
}
