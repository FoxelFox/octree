#import "../data/context.wgsl"

struct Octree {
    data: atomic<u32>,
    childs: array<atomic<u32>, 8>
}

const INVALID_INDEX: u32 = 0xFFFFFFFFu;

@group(0) @binding(0) var <uniform> context: Context;
@group(0) @binding(1) var<storage, read_write> nodes: array<Octree>;
@group(1) @binding(0) var prevFrameTexture: texture_2d<f32>;
@group(1) @binding(1) var smpler: sampler;

@vertex
fn main_vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  let c = context;
  const pos = array(
    vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
    vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
  );
  return vec4(pos[i], 0.0, 1.0);
}

fn traverse_octree(world_pos: vec3<f32>, grid_size: u32) -> u32 {
    var current_index = 0u;
    var pos = vec3<u32>(world_pos * f32(grid_size));
    
    // Clamp position to grid bounds
    pos = min(pos, vec3<u32>(grid_size - 1u));
    
    var depth = 0u;
    
    while (depth < context.max_depth) {
        let level_size = grid_size >> depth;
        if (level_size <= 1u) {
            break;
        }
        
        let half_size = level_size >> 1u;
        let relative_pos = vec3<u32>(
            pos.x % level_size,
            pos.y % level_size,
            pos.z % level_size
        );
        
        let octant_x = select(0u, 1u, relative_pos.x >= half_size);
        let octant_y = select(0u, 2u, relative_pos.y >= half_size);
        let octant_z = select(0u, 4u, relative_pos.z >= half_size);
        let octant = octant_x | octant_y | octant_z;
        
        let child_index = atomicLoad(&nodes[current_index].childs[octant]);
        if (child_index == INVALID_INDEX) {
            break;
        }
        
        current_index = child_index;
        depth = depth + 1u;
    }
    
    return atomicLoad(&nodes[current_index].data);
}

fn raycast_octree(ray_origin: vec3<f32>, ray_dir: vec3<f32>, grid_size: u32) -> vec3<f32> {
    let max_steps = 256u;
    let grid_size_f = f32(grid_size);
    let step_size = 0.5; // Smaller step size for better accuracy
    
    var t = 0.0;
    let max_distance = 50.0; // Reasonable max ray distance
    
    for (var step = 0u; step < max_steps; step = step + 1u) {
        let pos = ray_origin + ray_dir * t;
        
        // Check bounds in octree world space [0, grid_size]
        if (all(pos >= vec3<f32>(0.0)) && all(pos < vec3<f32>(grid_size_f))) {
            // Normalize position to [0, 1] for traverse_octree
            let normalized_pos = pos / grid_size_f;
            let voxel_data = traverse_octree(normalized_pos, grid_size);
            if (voxel_data > 0u) {
                return pos;
            }
        }
        
        t = t + step_size;
        if (t > max_distance) {
            break;
        }
    }
    
    return vec3<f32>(-1.0);
}

struct FragmentOutput {
  @location(0) colorForCanvas: vec4<f32>,
  @location(1) colorForTexture: vec4<f32>,
};

@fragment
fn main_fs(@builtin(position) pos: vec4<f32>) -> FragmentOutput {
  let uv = pos.xy / context.resolution;
  let ndc = vec4<f32>((uv - 0.5) * 2.0 * vec2<f32>(1.0, -1.0), 0.0, 1.0);
  
  // Transform from NDC to world space to get ray
  let view_pos = context.inverse_perspective * ndc;
  var world_pos = context.inverse_view * vec4<f32>(view_pos.xyz / view_pos.w, 1.0);
  
  // Camera position in world space
  var camera_pos = (context.inverse_view * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;

  // Create ray direction
  let ray_dir = normalize(world_pos.xyz - camera_pos);
  
  // Raycast through octree
  let hit_pos = raycast_octree(camera_pos, ray_dir, context.grid_size);
  
  var color: vec4<f32>;
  if (hit_pos.x >= 0.0) {
    // Hit something in the octree
    let depth = distance(camera_pos, hit_pos);
    let fog = 1.0 - exp(-depth * 0.1);
    color = vec4<f32>(0.2, 1.0, 0.2, 1.0) * (1.0 - fog) + vec4<f32>(0.1, 0.1, 0.2, 1.0) * fog;
  } else {
    // Debug: Show ray direction as color
    color = vec4<f32>(abs(ray_dir) * 0.5 + 0.5, 1.0);
  }

  // Mouse interaction
  var ar = context.resolution.x / context.resolution.y;
  var p = (uv - 0.5) * 2.0;
  p.x *= ar;
  var mouse = context.mouse_rel;
  mouse.x *= ar;
  let mouse_dist = distance(p, mouse);
  if (mouse_dist < 0.5) {
    color = vec4<f32>(0, 0, 0, 0);
  }

  let last = textureSample(prevFrameTexture, smpler, uv);
  let fade_last = max(vec4<f32>(0.0), last - 0.002);

  var output: FragmentOutput;
  output.colorForTexture = mix(fade_last, color, 0.1);
  output.colorForCanvas = color;
  return output;
}


