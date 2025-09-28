import {GPUContext} from "./gpu";
import {ContextUniform} from "./data/context";
import {TimingDisplay} from "./ui/timing-display";
import {FrameGraphManager} from "./ui/frame-graph-manager";
import {Streaming} from "./chunk/streaming";

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

const streaming = new Streaming();

// --- UI Components ---
const timingDisplay = new TimingDisplay();
const frameGraphManager = new FrameGraphManager();

function runOneTimeSetup() {
	gpu.update();

	for (const uniform of uniforms) {
		uniform.update();
	}

	streaming.init();
}

runOneTimeSetup();
loop();

// this has to be set after first render loop due to safari bug
document
	.getElementsByTagName("canvas")[0]
	.setAttribute("style", "position: fixed;");

function loop() {
	const frameStart = performance.now();

	gpu.update();

	// Handle voxel editing when pointer is locked and in Block mode
	if (streaming.voxelEditor && gpu.mouse.locked) {
		streaming.voxelEditorHandler.handleVoxelEditing();
	}

	for (const uniform of uniforms) {
		uniform.update();
	}

	const updateEncoder = device.createCommandEncoder();

	streaming.update(updateEncoder);

	// Get current renderer's timing based on pipeline mode
	const currentRenderTime = streaming.block.renderTime;

	// Update frame graph with GPU render time from active renderer
	frameGraphManager.getFrameGraph().addFrameTime(currentRenderTime);
	frameGraphManager.render();

	// Calculate CPU frame time (excludes GPU work) - moved to capture all CPU work
	const frameEnd = performance.now();
	const cpuFrameTime = frameEnd - frameStart;

	// Update timing display
	const stats = frameGraphManager.getFrameGraph().getCurrentStats();

	timingDisplay.update(
		currentRenderTime,
		streaming.light.renderTime,
		cpuFrameTime,
		stats,
		streaming.cull.count,
	);

	requestAnimationFrame(loop);
}
