// Note: We define COMPRESSION locally since we don't use the full context import
const COMPRESSION = 8;

// WebGPU DrawIndexedIndirect command format
struct Command {
	indexCount: u32,
	instanceCount: u32,
	firstIndex: u32,
	baseVertex: i32,
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

// Neighbor resolutions for LOD-aware sampling
struct NeighborLODs {
    // Store resolution of each neighbor (0 means no neighbor)
    // Order: -X, +X, -Y, +Y (in first vec4)
    //        -Z, +Z, unused, unused (in second vec4)
    // Using vec4<u32> for 16-byte alignment required by uniforms
    resolutions_0: vec4<u32>, // -X, +X, -Y, +Y
    resolutions_1: vec4<u32>, // -Z, +Z, unused, unused
}

// Context
@group(2) @binding(0) var<uniform> chunk_world_pos: vec4<i32>;
@group(2) @binding(1) var<uniform> chunk_resolution: vec4<u32>; // vec4 for alignment, only .x is used
@group(2) @binding(2) var<uniform> neighbor_lods: NeighborLODs;

// Helper to get chunk's meshlet count
fn chunk_meshlet_count() -> u32 {
    return chunk_resolution.x / u32(COMPRESSION);
}

// Convert world position to chunk-local compressed grid coordinates
fn worldToCompressedGrid(world_pos: vec3<f32>) -> vec3<f32> {
    let chunk_local_pos = world_pos - vec3<f32>(chunk_world_pos.xyz);
    // Each meshlet covers (256 / meshlet_count) world units
    let meshlet_count = chunk_meshlet_count();
    let meshlet_world_size = 256.0 / f32(meshlet_count);
    return chunk_local_pos / meshlet_world_size;
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
    return chunk_meshlet_count();
}

fn lightSegmentStride() -> u32 {
    let size = chunk_meshlet_count();
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

// Helper to convert 3D position to 1D index for a neighbor chunk with different resolution
fn to1D_neighbor(pos: vec3<i32>, neighbor_resolution: u32) -> u32 {
    let size = neighbor_resolution / u32(COMPRESSION);
    return u32(pos.z) * size * size + u32(pos.y) * size + u32(pos.x);
}

// Scale coordinates from current chunk space to neighbor chunk space
// If neighbor has coarser LOD (lower resolution), coordinates need to be scaled down
// If neighbor has finer LOD (higher resolution), coordinates need to be scaled up
fn scale_coord_to_neighbor(coord: i32, current_size: i32, neighbor_size: i32) -> i32 {
    // Scale factor: if current is 32 and neighbor is 16, scale = 0.5
    // coord in [0, 32) maps to [0, 16)
    return (coord * neighbor_size) / current_size;
}

fn sampleLightSegment(segment: u32, coords: vec3<i32>, neighbor_resolution: u32) -> vec2<f32> {
    // Calculate the actual grid size for this segment
    let segment_grid_size = i32(neighbor_resolution / u32(COMPRESSION));
    let max_index = segment_grid_size - 1;
    let clamped = clampVec3(coords, 0, max_index);

    // Use current chunk's stride for segment offset (all segments use same stride in combined buffer)
    let stride = lightSegmentStride();

    // Calculate index within the segment using the segment's actual size
    let size_sq = segment_grid_size * segment_grid_size;
    let index_i = clamped.z * size_sq + clamped.y * segment_grid_size + clamped.x;
    return light_data[segment * stride + u32(index_i)];
}

// Get light data at specific grid coordinates, sampling neighbor chunks when needed
fn getLightDataAtGrid(grid_pos: vec3<i32>) -> vec2<f32> {
    let size = i32(lightGridSize());

    // Sample from -X neighbor
    if (grid_pos.x < 0) {
        let neighbor_res = neighbor_lods.resolutions_0.x;
        if (neighbor_res == 0u) {
            return NO_LIGHT; // No neighbor
        }
        let neighbor_size = i32(neighbor_res / u32(COMPRESSION));
        // Scale coordinates to neighbor's resolution
        let scaled_y = scale_coord_to_neighbor(grid_pos.y, size, neighbor_size);
        let scaled_z = scale_coord_to_neighbor(grid_pos.z, size, neighbor_size);
        let neighbor_pos = vec3<i32>(neighbor_size - 1, scaled_y, scaled_z);
        // sampleLightSegment will clamp coordinates internally
        return sampleLightSegment(SEGMENT_NX, neighbor_pos, neighbor_res);
    }

    // Sample from +X neighbor
    if (grid_pos.x >= size) {
        let neighbor_res = neighbor_lods.resolutions_0.y;
        if (neighbor_res == 0u) {
            return NO_LIGHT;
        }
        let neighbor_size = i32(neighbor_res / u32(COMPRESSION));
        let scaled_y = scale_coord_to_neighbor(grid_pos.y, size, neighbor_size);
        let scaled_z = scale_coord_to_neighbor(grid_pos.z, size, neighbor_size);
        let neighbor_pos = vec3<i32>(0, scaled_y, scaled_z);
        return sampleLightSegment(SEGMENT_PX, neighbor_pos, neighbor_res);
    }

    // Sample from -Y neighbor
    if (grid_pos.y < 0) {
        let neighbor_res = neighbor_lods.resolutions_0.z;
        if (neighbor_res == 0u) {
            return NO_LIGHT;
        }
        let neighbor_size = i32(neighbor_res / u32(COMPRESSION));
        let scaled_x = scale_coord_to_neighbor(grid_pos.x, size, neighbor_size);
        let scaled_z = scale_coord_to_neighbor(grid_pos.z, size, neighbor_size);
        let neighbor_pos = vec3<i32>(scaled_x, neighbor_size - 1, scaled_z);
        return sampleLightSegment(SEGMENT_NY, neighbor_pos, neighbor_res);
    }

    // Sample from +Y neighbor
    if (grid_pos.y >= size) {
        let neighbor_res = neighbor_lods.resolutions_0.w;
        if (neighbor_res == 0u) {
            return NO_LIGHT;
        }
        let neighbor_size = i32(neighbor_res / u32(COMPRESSION));
        let scaled_x = scale_coord_to_neighbor(grid_pos.x, size, neighbor_size);
        let scaled_z = scale_coord_to_neighbor(grid_pos.z, size, neighbor_size);
        let neighbor_pos = vec3<i32>(scaled_x, 0, scaled_z);
        return sampleLightSegment(SEGMENT_PY, neighbor_pos, neighbor_res);
    }

    // Sample from -Z neighbor
    if (grid_pos.z < 0) {
        let neighbor_res = neighbor_lods.resolutions_1.x;
        if (neighbor_res == 0u) {
            return NO_LIGHT;
        }
        let neighbor_size = i32(neighbor_res / u32(COMPRESSION));
        let scaled_x = scale_coord_to_neighbor(grid_pos.x, size, neighbor_size);
        let scaled_y = scale_coord_to_neighbor(grid_pos.y, size, neighbor_size);
        let neighbor_pos = vec3<i32>(scaled_x, scaled_y, neighbor_size - 1);
        return sampleLightSegment(SEGMENT_NZ, neighbor_pos, neighbor_res);
    }

    // Sample from +Z neighbor
    if (grid_pos.z >= size) {
        let neighbor_res = neighbor_lods.resolutions_1.y;
        if (neighbor_res == 0u) {
            return NO_LIGHT;
        }
        let neighbor_size = i32(neighbor_res / u32(COMPRESSION));
        let scaled_x = scale_coord_to_neighbor(grid_pos.x, size, neighbor_size);
        let scaled_y = scale_coord_to_neighbor(grid_pos.y, size, neighbor_size);
        let neighbor_pos = vec3<i32>(scaled_x, scaled_y, 0);
        return sampleLightSegment(SEGMENT_PZ, neighbor_pos, neighbor_res);
    }

    // Sample from current chunk
    return sampleLightSegment(SEGMENT_CURRENT, grid_pos, chunk_resolution.x);
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
    // With indexed rendering, vertexCounts contains the actual unique vertex count
    let vertex_count = declared_vertex_count;

    if (vertex_count == 0u) {
        return;
    }

    // Early exit if this vertex slot is beyond the meshlet's actual vertex count
    if (vertex_in_meshlet >= vertex_count) {
        return;
    }

    // Calculate actual vertex index in the vertex buffer
    // baseVertex is the offset to the first vertex for this meshlet
    let vertex_index = u32(command.baseVertex) + vertex_in_meshlet;
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
