#import "../data/context.wgsl"

struct Mesh {
	vertexCount: u32,
	vertices: array<vec4<f32>, 1536>, // worst case is way larger than 2048
}

  struct Command {
      vertexCount: u32,
      instanceCount: u32,
      firstVertex: u32,
      firstInstance: u32,
  }

// Input
@group(0) @binding(0) var<storage, read> voxel: array<f32>;
@group(1) @binding(0) var<uniform> context: Context;

// Output
@group(0) @binding(1) var<storage, read_write> meshes: array<Mesh>;
@group(0) @binding(2) var<storage, read_write> commands: array<Command>;


// All 36 vertices for a cube's 12 triangles, with correct CCW winding.
const CUBE_TRIANGLE_VERTICES = array<vec3<u32>, 36>(
	// +X (right)
	vec3(1, 0, 1), vec3(1, 0, 0), vec3(1, 1, 0),
	vec3(1, 0, 1), vec3(1, 1, 0), vec3(1, 1, 1),
	// -X (left)
	vec3(0, 0, 0), vec3(0, 0, 1), vec3(0, 1, 1),
	vec3(0, 0, 0), vec3(0, 1, 1), vec3(0, 1, 0),
	// +Y (top)
	vec3(0, 1, 1), vec3(1, 1, 0), vec3(0, 1, 0),
	vec3(0, 1, 1), vec3(1, 1, 1), vec3(1, 1, 0),
	// -Y (bottom)
	vec3(0, 0, 0), vec3(1, 0, 1), vec3(0, 0, 1),
	vec3(0, 0, 0), vec3(1, 0, 0), vec3(1, 0, 1),
	// +Z (front)
	vec3(0, 0, 1), vec3(1, 1, 1), vec3(0, 1, 1),
	vec3(0, 0, 1), vec3(1, 0, 1), vec3(1, 1, 1),
	// -Z (back)
	vec3(1, 0, 0), vec3(0, 1, 0), vec3(1, 1, 0),
	vec3(1, 0, 0), vec3(0, 0, 0), vec3(0, 1, 0)
);

const NEIGHBORS = array<vec3<i32>, 6>(
	vec3(1, 0, 0), vec3(-1, 0, 0),
	vec3(0, 1, 0), vec3(0, -1, 0),
	vec3(0, 0, 1), vec3(0, 0, -1)
);

const COMPRESSION = 8;

fn to1D(id: vec3<u32>) -> u32 {
	return id.z * context.grid_size * context.grid_size + id.y * context.grid_size + id.x;
}

fn to1DSmall(id: vec3<u32>) -> u32 {
	let size = context.grid_size / COMPRESSION;
	return id.z * size * size + id.y * size + id.x;
}

fn getVoxel(pos: vec3<u32>) -> f32 {
	let index = pos.z * context.grid_size * context.grid_size + pos.y * context.grid_size + pos.x;
	return voxel[index];
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {

	var mesh = Mesh();
	var command = Command();
	mesh.vertexCount = 0;




	for (var x = 0u; x < COMPRESSION; x++) {
		for (var y = 0u; y < COMPRESSION; y++) {
			for (var z = 0u; z < COMPRESSION; z++) {
				let coord = vec3<u32>(x,y,z);
				let voxel = getVoxel(coord + id * COMPRESSION);
				if (voxel <= 0.0) {
					for (var n = 0; n < 6; n++) {
						// for every neighbor face
						let neighborPos = vec3<i32>(coord + id * COMPRESSION) + NEIGHBORS[n];
						var shouldAddFace = false;
						
						// Check if neighbor is out of bounds (edge of grid)
						if (neighborPos.x < 0 || neighborPos.y < 0 || neighborPos.z < 0 ||
						    neighborPos.x >= i32(context.grid_size) ||
						    neighborPos.y >= i32(context.grid_size) ||
						    neighborPos.z >= i32(context.grid_size)) {
							shouldAddFace = true; // Out of bounds = add face
						} else {
							// In bounds, check if neighbor is empty
							shouldAddFace = getVoxel(vec3<u32>(neighborPos)) > 0.0;
						}
						
						if (shouldAddFace) {
							// Generate 6 vertices (2 triangles) for this face
							for (var v = 0; v < 6; v++) {
								var vertexOffset = CUBE_TRIANGLE_VERTICES[n * 6 + v];
								var worldVertex = vec3<f32>(coord + id * COMPRESSION) + vec3<f32>(vertexOffset);
								mesh.vertices[mesh.vertexCount] = vec4<f32>(worldVertex, 1.0);
								mesh.vertexCount++;
							}
						}
					}
				}
			}
		}
	}


	let index = to1DSmall(id);

	meshes[index] = mesh;

	command.vertexCount = mesh.vertexCount;
	command.instanceCount = 1u;
	command.firstVertex = 0u;
	command.firstInstance = index;
	commands[index] = command;

}