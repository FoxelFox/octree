enable f16;

#import "../../data/context.wgsl"


struct Command {
	vertexCount: u32,
	instanceCount: u32,
	firstVertex: u32,
	firstInstance: u32,
}

// Voxel data structure containing density and color
struct VoxelData {
    density: f32,
    color: u32, // Packed RGBA color
}

// Input
@group(0) @binding(0) var<storage, read> voxels: array<VoxelData>;
@group(1) @binding(0) var<uniform> context: Context;
@group(1) @binding(1) var<uniform> offset: vec3<u32>;

// Output
@group(0) @binding(1) var<storage, read_write> vertexCounts: array<u32>;
@group(0) @binding(2) var<storage, read_write> vertices: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read_write> normals: array<vec3<f16>>;
@group(0) @binding(4) var<storage, read_write> colors: array<u32>;
@group(0) @binding(5) var<storage, read_write> commands: array<Command>;
@group(0) @binding(6) var<storage, read_write> density: array<u32>;


// Marching cubes edge table uniform buffer (vec4<u32> for 16-byte alignment)
@group(1) @binding(2) var<uniform> edgeTable: array<vec4<u32>, 64>;

// Marching cubes triangle table uniform buffer (vec4<i32> for 16-byte alignment)
@group(1) @binding(3) var<uniform> triangleTable: array<vec4<i32>, 1024>;

// Cube vertex positions (8 corners of a unit cube)
const CUBE_VERTICES = array<vec3<f32>, 8>(
	vec3(0.0, 0.0, 0.0), // 0
	vec3(1.0, 0.0, 0.0), // 1
	vec3(1.0, 1.0, 0.0), // 2
	vec3(0.0, 1.0, 0.0), // 3
	vec3(0.0, 0.0, 1.0), // 4
	vec3(1.0, 0.0, 1.0), // 5
	vec3(1.0, 1.0, 1.0), // 6
	vec3(0.0, 1.0, 1.0)  // 7
);

// Edge connections for interpolation
const EDGE_VERTICES = array<array<u32, 2>, 12>(
	array(0u, 1u), array(1u, 2u), array(2u, 3u), array(3u, 0u),
	array(4u, 5u), array(5u, 6u), array(6u, 7u), array(7u, 4u),
	array(0u, 4u), array(1u, 5u), array(2u, 6u), array(3u, 7u)
);

fn calculateGradient(pos: vec3<i32>) -> vec3<f32> {
	let dx = getVoxelDensitySafe(pos + vec3(1, 0, 0)) - getVoxelDensitySafe(pos - vec3(1, 0, 0));
	let dy = getVoxelDensitySafe(pos + vec3(0, 1, 0)) - getVoxelDensitySafe(pos - vec3(0, 1, 0));
	let dz = getVoxelDensitySafe(pos + vec3(0, 0, 1)) - getVoxelDensitySafe(pos - vec3(0, 0, 1));

	let gradient = vec3<f32>(dx, dy, dz);
	let length = length(gradient);
	if (length > 0.0001) {
		return -gradient / length;
	}
	return vec3<f32>(0.0, 1.0, 0.0); // Default normal if gradient is zero
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

// Interpolate colors for marching cubes edges
fn interpolateColor(color1: u32, color2: u32, val1: f32, val2: f32) -> u32 {
    let isolevel = 0.0;
    if (abs(isolevel - val1) < 0.00001) {
        return color1;
    }
    if (abs(isolevel - val2) < 0.00001) {
        return color2;
    }
    if (abs(val1 - val2) < 0.00001) {
        return color1;
    }

    let mu = (isolevel - val1) / (val2 - val1);
    let c1 = unpackColor(color1);
    let c2 = unpackColor(color2);
    let interpolated = c1 + mu * (c2 - c1);
    return packColor(interpolated);
}

fn getVoxelData(pos: vec3<u32>) -> VoxelData {
	let index = pos.z * context.grid_size * context.grid_size + pos.y * context.grid_size + pos.x;
	return voxels[index];
}

fn getVoxelDensity(pos: vec3<u32>) -> f32 {
	return getVoxelData(pos).density;
}

fn getVoxelColor(pos: vec3<u32>) -> u32 {
	return getVoxelData(pos).color;
}

fn getVoxelDensitySafe(pos: vec3<i32>) -> f32 {
	if (pos.x < 0 || pos.y < 0 || pos.z < 0 ||
	    pos.x >= i32(context.grid_size) ||
	    pos.y >= i32(context.grid_size) ||
	    pos.z >= i32(context.grid_size)) {
		return 1.0; // Outside bounds = solid
	}
	return getVoxelDensity(vec3<u32>(pos));
}

fn getVoxelColorSafe(pos: vec3<i32>) -> u32 {
	if (pos.x < 0 || pos.y < 0 || pos.z < 0 ||
	    pos.x >= i32(context.grid_size) ||
	    pos.y >= i32(context.grid_size) ||
	    pos.z >= i32(context.grid_size)) {
		return 0x808080FFu; // Default gray color for outside bounds
	}
	return getVoxelColor(vec3<u32>(pos));
}

fn interpolateVertex(p1: vec3<f32>, p2: vec3<f32>, val1: f32, val2: f32) -> vec3<f32> {
	let isolevel = 0.0;
	if (abs(isolevel - val1) < 0.00001) {
		return p1;
	}
	if (abs(isolevel - val2) < 0.00001) {
		return p2;
	}
	if (abs(val1 - val2) < 0.00001) {
		return p1;
	}

	let mu = (isolevel - val1) / (val2 - val1);
	return p1 + mu * (p2 - p1);
}

fn interpolateNormal(n1: vec3<f32>, n2: vec3<f32>, val1: f32, val2: f32) -> vec3<f32> {
	let isolevel = 0.0;
	if (abs(isolevel - val1) < 0.00001) {
		return normalize(n1);
	}
	if (abs(isolevel - val2) < 0.00001) {
		return normalize(n2);
	}
	if (abs(val1 - val2) < 0.00001) {
		return normalize(n1);
	}

	let mu = (isolevel - val1) / (val2 - val1);
	return normalize(n1 + mu * (n2 - n1));
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {

	var vertexCount = 0u;
	var command = Command();
	let actualId = id + offset;
	let sSize = context.grid_size / COMPRESSION;
	let index = to1DSmall(actualId);
	density[index] = 0;

	for (var x = 0u; x < COMPRESSION; x++) {
		for (var y = 0u; y < COMPRESSION; y++) {
			for (var z = 0u; z < COMPRESSION; z++) {
				let coord = vec3<u32>(x, y, z);
				let worldPos = vec3<i32>(coord + actualId * COMPRESSION);

				// Get the 8 corner values and colors of the cube
				var cubeValues: array<f32, 8>;
				var cubeColors: array<u32, 8>;
				cubeValues[0] = getVoxelDensitySafe(worldPos + vec3(0, 0, 0));
				cubeValues[1] = getVoxelDensitySafe(worldPos + vec3(1, 0, 0));
				cubeValues[2] = getVoxelDensitySafe(worldPos + vec3(1, 1, 0));
				cubeValues[3] = getVoxelDensitySafe(worldPos + vec3(0, 1, 0));
				cubeValues[4] = getVoxelDensitySafe(worldPos + vec3(0, 0, 1));
				cubeValues[5] = getVoxelDensitySafe(worldPos + vec3(1, 0, 1));
				cubeValues[6] = getVoxelDensitySafe(worldPos + vec3(1, 1, 1));
				cubeValues[7] = getVoxelDensitySafe(worldPos + vec3(0, 1, 1));

				cubeColors[0] = getVoxelColorSafe(worldPos + vec3(0, 0, 0));
				cubeColors[1] = getVoxelColorSafe(worldPos + vec3(1, 0, 0));
				cubeColors[2] = getVoxelColorSafe(worldPos + vec3(1, 1, 0));
				cubeColors[3] = getVoxelColorSafe(worldPos + vec3(0, 1, 0));
				cubeColors[4] = getVoxelColorSafe(worldPos + vec3(0, 0, 1));
				cubeColors[5] = getVoxelColorSafe(worldPos + vec3(1, 0, 1));
				cubeColors[6] = getVoxelColorSafe(worldPos + vec3(1, 1, 1));
				cubeColors[7] = getVoxelColorSafe(worldPos + vec3(0, 1, 1));


				if (cubeValues[0] < 0.0) {
					density[index]++;
				}


				// Calculate cube configuration index
				var cubeIndex = 0u;
				if (cubeValues[0] < 0.0) { cubeIndex |= 1u; }
				if (cubeValues[1] < 0.0) { cubeIndex |= 2u; }
				if (cubeValues[2] < 0.0) { cubeIndex |= 4u; }
				if (cubeValues[3] < 0.0) { cubeIndex |= 8u; }
				if (cubeValues[4] < 0.0) { cubeIndex |= 16u; }
				if (cubeValues[5] < 0.0) { cubeIndex |= 32u; }
				if (cubeValues[6] < 0.0) { cubeIndex |= 64u; }
				if (cubeValues[7] < 0.0) { cubeIndex |= 128u; }

				// Skip empty cubes
				if (cubeIndex == 0u || cubeIndex == 255u) {
					continue;
				}



				// Get edge configuration
				let edgeVec4Index = cubeIndex / 4u;
				let edgeComponent = cubeIndex % 4u;
				let edges = edgeTable[edgeVec4Index][edgeComponent];
				if (edges == 0u) {
					continue;
				}

				// Calculate interpolated vertices, normals, and colors on edges
				var vertexList: array<vec3<f32>, 12>;
				var normalList: array<vec3<f32>, 12>;
				var colorList: array<u32, 12>;

				// Check each edge bit and interpolate if necessary
				for (var i = 0u; i < 12u; i++) {
					let edgeBit = 1u << i;
					if ((edges & edgeBit) != 0u) {
						let v1 = EDGE_VERTICES[i][0];
						let v2 = EDGE_VERTICES[i][1];
						let p1 = vec3<f32>(worldPos) + CUBE_VERTICES[v1];
						let p2 = vec3<f32>(worldPos) + CUBE_VERTICES[v2];
						vertexList[i] = interpolateVertex(p1, p2, cubeValues[v1], cubeValues[v2]);

						// Calculate gradients at cube vertices
						let n1 = calculateGradient(worldPos + vec3<i32>(CUBE_VERTICES[v1]));
						let n2 = calculateGradient(worldPos + vec3<i32>(CUBE_VERTICES[v2]));
						normalList[i] = interpolateNormal(n1, n2, cubeValues[v1], cubeValues[v2]);

						// Use hard color edges - pick the color from the "inside" vertex (negative density)
						if (cubeValues[v1] < cubeValues[v2]) {
							colorList[i] = cubeColors[v1]; // v1 is more "inside"
						} else {
							colorList[i] = cubeColors[v2]; // v2 is more "inside"
						}
					}
				}

				// Generate triangles using lookup table
				let baseTriangleIndex = cubeIndex * 16u;
				// Access triangleTable as vec4 arrays
				for (var i = 0u; i < 16u; i += 3u) {
					let idx1 = baseTriangleIndex + i;
					let idx2 = baseTriangleIndex + i + 1u;
					let idx3 = baseTriangleIndex + i + 2u;
					
					let edge1 = triangleTable[idx1 / 4u][idx1 % 4u];
					let edge2 = triangleTable[idx2 / 4u][idx2 % 4u];
					let edge3 = triangleTable[idx3 / 4u][idx3 % 4u];
					
					if (edge1 == -1) {
						break;
					}

					// Calculate global vertex indices
					let baseVertexIndex = index * 1536u + vertexCount;

					// Ensure we don't exceed vertex buffer capacity
					if (vertexCount + 3 <= 1536 && edge1 >= 0 && edge2 >= 0 && edge3 >= 0) {
						let v1 = vertexList[edge1];
						let v2 = vertexList[edge2];
						let v3 = vertexList[edge3];

						let n1 = normalList[edge1];
						let n2 = normalList[edge2];
						let n3 = normalList[edge3];

						let c1 = colorList[edge1];
						let c2 = colorList[edge2];
						let c3 = colorList[edge3];

						vertices[baseVertexIndex] = vec4<f16>(vec3<f16>(v1), 1.0h);
						vertices[baseVertexIndex + 1] = vec4<f16>(vec3<f16>(v2), 1.0h);
						vertices[baseVertexIndex + 2] = vec4<f16>(vec3<f16>(v3), 1.0h);

						normals[baseVertexIndex] = vec3<f16>(n1);
						normals[baseVertexIndex + 1] = vec3<f16>(n2);
						normals[baseVertexIndex + 2] = vec3<f16>(n3);

						colors[baseVertexIndex] = c1;
						colors[baseVertexIndex + 1] = c2;
						colors[baseVertexIndex + 2] = c3;

						vertexCount += 3u;
					} else if (edge1 < 0 || edge2 < 0 || edge3 < 0) {
						break; // End of triangles for this configuration
					}
				}
			}
		}
	}

	vertexCounts[index] = vertexCount;

	command.vertexCount = vertexCount;
	command.instanceCount = 1u;
	command.firstVertex = 0u;
	command.firstInstance = index;
	commands[index] = command;

}
