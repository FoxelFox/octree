import { GPUContext } from "./gpu";
import { ContextUniform } from "./data/context";
import { Noise } from "./pipeline/generation/noise";
import { Block } from "./pipeline/rendering/block";
import { Mesh } from "./pipeline/generation/mesh";
import { Cull } from "./pipeline/rendering/cull";
import { Light } from "./pipeline/rendering/light";
import { VoxelEditor } from "./pipeline/generation/voxel_editor";
import { TimingDisplay } from "./ui/timing-display";
import { FrameGraphManager } from "./ui/frame-graph-manager";
import { VoxelEditorHandler } from "./ui/voxel-editor";

export const gpu = new GPUContext();
await gpu.init();

export const maxDepth = 8;
export const gridSize = Math.pow(2, maxDepth);
export const worstCaseMaxNodes = Math.floor(
	(Math.pow(8, maxDepth + 1) - 1) / 7,
);
export const device = gpu.device;
export const context = gpu.context;
export const canvas = gpu.canvas;
export const mouse = gpu.mouse;
export const camera = gpu.camera;
export const time = gpu.time;
export const compression = 8;
export const contextUniform = new ContextUniform();

console.log("maxDepth:", maxDepth);
console.log("gridSize:", gridSize);

const uniforms = [contextUniform];

// --- Setup Pipelines ---
const noise = new Noise();
const light = new Light();
const block = new Block();
const mesh = new Mesh();
const cull = new Cull();
block.mesh = mesh;
block.cull = cull;
block.light = light;

// --- UI Components ---
const timingDisplay = new TimingDisplay();
const frameGraphManager = new FrameGraphManager();
let voxelEditorHandler: VoxelEditorHandler;

// --- Voxel Editor ---
let voxelEditor: VoxelEditor;

async function runOneTimeSetup() {
	gpu.update();

	for (const uniform of uniforms) {
		uniform.update();
	}

	console.log("Running one-time octree generation and compaction...");
	const setupEncoder = device.createCommandEncoder();
	noise.update(setupEncoder);

	device.queue.submit([setupEncoder.finish()]);

	mesh.init(noise);
	cull.init(noise, mesh);
	light.init(mesh);

	// Generate distance field from voxel data
	console.log("Generating distance field...");
	const encoder = device.createCommandEncoder();

	mesh.update(encoder);
	light.update(encoder, mesh);
	device.queue.submit([encoder.finish()]);

	mesh.afterUpdate();
	light.afterUpdate();

	// Initialize voxel editor after all systems are ready
	voxelEditor = new VoxelEditor(block, noise, mesh, cull, light);
	voxelEditorHandler = new VoxelEditorHandler(gpu, voxelEditor);

	console.log("Setup complete.");
}

await runOneTimeSetup();
loop();

// this has to be set after first render loop due to safari bug
document
	.getElementsByTagName("canvas")[0]
	.setAttribute("style", "position: fixed;");

function loop() {
	const frameStart = performance.now();

	gpu.update();

	// Handle voxel editing when pointer is locked and in Block mode
	if (voxelEditor && gpu.mouse.locked) {
		voxelEditorHandler.handleVoxelEditing();
	}

	for (const uniform of uniforms) {
		uniform.update();
	}

	const updateEncoder = device.createCommandEncoder();

	// Update lighting and culling every frame (async on GPU)
	light.update(updateEncoder, mesh);
	cull.update(updateEncoder);
	block.update(updateEncoder);

	device.queue.submit([updateEncoder.finish()]);

	block.afterUpdate();
	mesh.afterUpdate();
	cull.afterUpdate();
	light.afterUpdate();

	// Get current renderer's timing based on pipeline mode
	const currentRenderTime = block.renderTime;

	// Update frame graph with GPU render time from active renderer
	frameGraphManager.getFrameGraph().addFrameTime(currentRenderTime);
	frameGraphManager.render();

	// Calculate CPU frame time (excludes GPU work) - moved to capture all CPU work
	const frameEnd = performance.now();
	const cpuFrameTime = frameEnd - frameStart;

	// Update timing display
	const stats = frameGraphManager.getFrameGraph().getCurrentStats();

	timingDisplay.update(currentRenderTime, light.renderTime, cpuFrameTime, stats, cull.count);

	requestAnimationFrame(loop);
}
