#import "../data/context.wgsl"


@group(0) @binding(3) var<storage, read_write> counter: atomic<u32>;
