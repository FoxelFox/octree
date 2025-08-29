struct Context {
  resolution: vec2<f32>,
  mouse_abs: vec2<f32>,
  mouse_rel: vec2<f32>,
  time: f32,
  delta: f32,
  grid_size: u32,
  max_depth: u32,
  view: mat4x4<f32>,
  inverse_view: mat4x4<f32>,
  perspective: mat4x4<f32>,
  inverse_perspective: mat4x4<f32>,
  prev_view_projection: mat4x4<f32>,
  jitter_offset: vec2<f32>,
  camera_velocity: vec3<f32>,
  frame_count: u32,
  render_mode: u32, // 0=normal, 1=heatmap, 2=sdf_normal, 3=sdf_heatmap, 4=hybrid_normal, 5=hybrid_heatmap
  random_seed: f32,
  // Distance field parameters
  sdf_epsilon: f32,
  sdf_max_steps: u32,
  sdf_over_relaxation: f32,
  // TAA toggle
  taa_enabled: u32,
  // Hybrid parameters
  hybrid_threshold: f32, // Node size threshold for switching to SDF
}

const COMPRESSION = 8;

fn to1D(id: vec3<u32>) -> u32 {
	return id.z * context.grid_size * context.grid_size + id.y * context.grid_size + id.x;
}

fn to1DSmall(id: vec3<u32>) -> u32 {
	let size = context.grid_size / COMPRESSION;
	return id.z * size * size + id.y * size + id.x;
}