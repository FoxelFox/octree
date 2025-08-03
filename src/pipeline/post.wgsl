#import "../data/context.wgsl"

@group(0) @binding(0) var <uniform> context: Context;
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

struct FragmentOutput {
  @location(0) colorForCanvas: vec4<f32>,
  @location(1) colorForTexture: vec4<f32>,
};

@fragment
fn main_fs(@builtin(position) pos: vec4<f32>) -> FragmentOutput {

  var p = (pos.xy / context.resolution - 0.5) * 2.0;
  var ar = context.resolution.x / context.resolution.y;
  p.x *= ar;

  var mouse = context.mouse_rel;
  mouse.x *= ar;

  var d = distance(p, mouse);
  if (d < 0.05) {
    d = 1.0;
  } else {
    d = 0.0;
  }

  var color = vec4(1.0,0.7,0.2, d);

  let uv = (pos.xy / context.resolution);

  var last = textureSample(prevFrameTexture, smpler, uv);
  last.a *= 0.9;

  var output: FragmentOutput;
  output.colorForTexture = mix(last, color, 0.5);
  output.colorForCanvas = mix(last, color, 0.5);
  return output;
}


