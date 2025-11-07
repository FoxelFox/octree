import {
	canvas,
	compression,
	context,
	contextUniform,
	device,
	gridSize,
} from "../../index";
import shader from "./block.wgsl" with { type: "text" };
import spaceBackgroundShader from "../generation/space_background.wgsl" with { type: "text" };
import { Chunk } from "../../chunk/chunk";

// Simple shader for rendering space background as full-screen quad
const spaceBackgroundQuadShader = `
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
  random_seed: f32,
  sdf_epsilon: f32,
  sdf_max_steps: u32,
  sdf_over_relaxation: f32,
  hybrid_threshold: f32,
}

@group(0) @binding(0) var<uniform> context: Context;
@group(1) @binding(0) var space_background_texture: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    // Full-screen quad
    const pos = array(
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
        vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
    );
    return vec4(pos[i], 0.0, 1.0);
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

// Sample space background with separate nebula and stars
fn sample_space_background(ray_dir: vec3<f32>) -> vec3<f32> {
    let uv = ray_direction_to_uv(ray_dir);
    let texture_size = textureDimensions(space_background_texture);
    let coord = vec2<i32>(uv * vec2<f32>(texture_size));
    let data = textureLoad(space_background_texture, coord, 0);

    let nebula = data.rgb;
    let star_intensity = data.a;

    // Reconstruct star color from intensity (approximate original colors)
    let star_color = vec3<f32>(1.0, 0.9, 0.8) * star_intensity;

    return nebula + star_color;
}

// Calculate ray direction from UV coordinates
fn calculate_ray_direction(uv: vec2<f32>, camera_pos: vec3<f32>) -> vec3<f32> {
    let ndc = vec4<f32>((uv - 0.5) * 2.0 * vec2<f32>(1.0, -1.0), 0.0, 1.0);
    let view_pos = context.inverse_perspective * ndc;
    let world_pos_bg = context.inverse_view * vec4<f32>(view_pos.xyz / view_pos.w, 1.0);
    return normalize(world_pos_bg.xyz - camera_pos);
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = frag_coord.xy / context.resolution;
    let camera_pos = context.inverse_view[3].xyz;
    let ray_dir = calculate_ray_direction(uv, camera_pos);
    let space_color = sample_space_background(ray_dir);
    return vec4<f32>(space_color, 1.0);
}
`;

export class Block {
	// Forward rendering pipeline
	forwardPipeline: GPURenderPipeline;
	forwardBindGroups = new Map<Chunk, GPUBindGroup>();
	forwardUniformBindGroup: GPUBindGroup;
	forwardSpaceBindGroup: GPUBindGroup;

	// Space background
	spaceBackgroundTexture: GPUTexture;
	spaceBackgroundPipeline: GPUComputePipeline;
	spaceBackgroundBindGroup: GPUBindGroup;
	spaceBackgroundQuadPipeline: GPURenderPipeline;
	spaceBackgroundQuadContextBindGroup: GPUBindGroup;
	spaceBackgroundQuadTextureBindGroup: GPUBindGroup;

	// Depth texture for forward rendering
	depthTexture: GPUTexture;

	initialized: boolean;

	constructor() {
		this.createDepthTexture();
		this.createSpaceBackgroundTexture();

		// Create space background compute pipeline
		this.spaceBackgroundPipeline = device.createComputePipeline({
			label: "Space Background Generation",
			layout: "auto",
			compute: {
				module: device.createShaderModule({
					label: "Space Background Compute Shader",
					code: spaceBackgroundShader,
				}),
			},
		});

		// Create space background bind group
		this.spaceBackgroundBindGroup = device.createBindGroup({
			label: "Space Background",
			layout: this.spaceBackgroundPipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: this.spaceBackgroundTexture.createView() },
			],
		});

		// Create space background quad rendering pipeline
		this.spaceBackgroundQuadPipeline = device.createRenderPipeline({
			label: "Space Background Quad",
			layout: "auto",
			fragment: {
				module: device.createShaderModule({
					label: "Space Background Quad Fragment Shader",
					code: spaceBackgroundQuadShader,
				}),
				targets: [{ format: "bgra8unorm" }],
			},
			vertex: {
				module: device.createShaderModule({
					label: "Space Background Quad Vertex Shader",
					code: spaceBackgroundQuadShader,
				}),
			},
			primitive: {
				topology: "triangle-list",
			},
		});

		// Create forward rendering pipeline
		this.forwardPipeline = device.createRenderPipeline({
			label: "Block Forward Rendering",
			layout: "auto",
			fragment: {
				module: device.createShaderModule({
					label: "Block Forward Fragment Shader",
					code: shader,
				}),
				targets: [{ format: "bgra8unorm" }],
			},
			vertex: {
				module: device.createShaderModule({
					label: "Block Forward Vertex Shader",
					code: shader,
				}),
			},
			primitive: {
				topology: "triangle-list",
				cullMode: "front",
			},
			depthStencil: {
				depthWriteEnabled: true,
				depthCompare: "less",
				format: "depth24plus",
			},
		});

		this.forwardUniformBindGroup = device.createBindGroup({
			label: "Block Forward Context",
			layout: this.forwardPipeline.getBindGroupLayout(1),
			entries: [{ binding: 0, resource: { buffer: contextUniform.uniformBuffer } }],
		});

		this.forwardSpaceBindGroup = device.createBindGroup({
			label: "Forward Space Background",
			layout: this.forwardPipeline.getBindGroupLayout(2),
			entries: [
				{ binding: 0, resource: this.spaceBackgroundTexture.createView() },
			],
		});

		// Create space background quad bind groups
		this.spaceBackgroundQuadContextBindGroup = device.createBindGroup({
			label: "Space Background Quad Context",
			layout: this.spaceBackgroundQuadPipeline.getBindGroupLayout(0),
			entries: [{ binding: 0, resource: { buffer: contextUniform.uniformBuffer } }],
		});

		this.spaceBackgroundQuadTextureBindGroup = device.createBindGroup({
			label: "Space Background Quad Texture",
			layout: this.spaceBackgroundQuadPipeline.getBindGroupLayout(1),
			entries: [
				{ binding: 0, resource: this.spaceBackgroundTexture.createView() },
			],
		});
	}

	createDepthTexture() {
		// Destroy existing texture
		if (this.depthTexture) this.depthTexture.destroy();

		const size = { width: canvas.width, height: canvas.height };

		// Depth texture
		this.depthTexture = device.createTexture({
			size,
			format: "depth24plus",
			usage: GPUTextureUsage.RENDER_ATTACHMENT,
		});
	}

	createSpaceBackgroundTexture() {
		// Destroy existing texture
		if (this.spaceBackgroundTexture) this.spaceBackgroundTexture.destroy();

		// Create space background texture (2048x1024 for good quality)
		this.spaceBackgroundTexture = device.createTexture({
			size: { width: 2048, height: 1024 },
			format: "rgba8unorm",
			usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
		});
	}

	generateSpaceBackground(commandEncoder: GPUCommandEncoder) {
		// Generate space background texture once using compute shader
		const computePass = commandEncoder.beginComputePass({
			label: "Generate Space Background",
		});

		computePass.setPipeline(this.spaceBackgroundPipeline);
		computePass.setBindGroup(0, this.spaceBackgroundBindGroup);
		computePass.dispatchWorkgroups(
			Math.ceil(2048 / 8), // workgroup size is 8x8
			Math.ceil(1024 / 8),
		);
		computePass.end();

		this.initialized = true;
	}

	update(commandEncoder: GPUCommandEncoder, chunks: Chunk[]) {
		if (!this.initialized) {
			// Generate space background texture once after initialization
			this.generateSpaceBackground(commandEncoder);
		}

		// Recreate depth texture if canvas size changed
		if (
			this.depthTexture.width !== canvas.width ||
			this.depthTexture.height !== canvas.height
		) {
			this.createDepthTexture();
		}

		// Pass 1: Render space background (full-screen quad)
		const backgroundPass = commandEncoder.beginRenderPass({
			label: "Space Background",
			colorAttachments: [
				{
					view: context.getCurrentTexture().createView(),
					loadOp: "clear",
					storeOp: "store",
					clearValue: { r: 0, g: 0, b: 0, a: 1 },
				},
			],
		});

		backgroundPass.setPipeline(this.spaceBackgroundQuadPipeline);
		backgroundPass.setBindGroup(0, this.spaceBackgroundQuadContextBindGroup);
		backgroundPass.setBindGroup(1, this.spaceBackgroundQuadTextureBindGroup);
		backgroundPass.draw(6); // Full-screen quad
		backgroundPass.end();

		// Pass 2: Forward render all chunks on top of background
		const forwardPass = commandEncoder.beginRenderPass({
			label: "Block Forward Rendering",
			colorAttachments: [
				{
					view: context.getCurrentTexture().createView(),
					loadOp: "load", // Load the background we just rendered
					storeOp: "store",
				},
			],
			depthStencilAttachment: {
				view: this.depthTexture.createView(),
				depthClearValue: 1.0,
				depthLoadOp: "clear",
				depthStoreOp: "store",
			},
		});

		forwardPass.setPipeline(this.forwardPipeline);

		// Render all chunks in a single pass
		for (const chunk of chunks) {
			forwardPass.setBindGroup(0, this.forwardBindGroups.get(chunk));
			forwardPass.setBindGroup(1, this.forwardUniformBindGroup);
			forwardPass.setBindGroup(2, this.forwardSpaceBindGroup);

			// Draw instances for each mesh chunk (only if culling data is ready)
			if (chunk.indicesArray && chunk.indicesArray.length > 0 && chunk.indicesBuffer) {
				// Set index buffer for indexed rendering
				forwardPass.setIndexBuffer(chunk.indicesBuffer, "uint32");

				const maxMeshIndex = Math.pow(gridSize / compression, 3) - 1;
				for (let i = 0; i < chunk.indicesArray.length; ++i) {
					const meshIndex = chunk.indicesArray[i];
					if (
						typeof meshIndex === "number" &&
						isFinite(meshIndex) &&
						meshIndex <= maxMeshIndex
					) {
						// Use indexed indirect draw (20 bytes per command: 5 u32s)
						// Command format: indexCount, instanceCount, firstIndex, baseVertex, firstInstance
						forwardPass.drawIndexedIndirect(chunk.commands, meshIndex * 20);
					} else if (meshIndex > maxMeshIndex) {
						console.warn(
							`Mesh index ${meshIndex} exceeds maximum ${maxMeshIndex}, skipping`,
						);
					}
				}
			}
		}

		forwardPass.end();
	}

	registerChunk(chunk: Chunk) {
		// Create forward rendering bind group
		const forwardBindGroup = device.createBindGroup({
			label: "Block Forward Meshes",
			layout: this.forwardPipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 1, resource: { buffer: chunk.vertices } },
				{ binding: 2, resource: { buffer: chunk.normals } },
				{ binding: 3, resource: { buffer: chunk.colors } },
			],
		});

		this.forwardBindGroups.set(chunk, forwardBindGroup);
	}

	unregisterChunk(chunk: Chunk) {
		this.forwardBindGroups.delete(chunk);
	}
}
