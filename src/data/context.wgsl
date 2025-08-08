struct Context {
  resolution: vec2<f32>,
  mouse_abs: vec2<f32>,
  mouse_rel: vec2<f32>,
  time: f32,
  delta: f32,
  view: mat4x4<f32>,
  inverse_view: mat4x4<f32>,
  perspective: mat4x4<f32>,
  inverse_perspective: mat4x4<f32>,
}