import { GPUContext } from "./gpu";
import { Post } from "./pipeline/post";
import { ContextUniform } from "./data/context";
import { Noise } from "./pipeline/noise";
import { DistanceField } from "./pipeline/distance_field";
import { FrameGraph } from "./ui/FrameGraph";
import { Block } from "./pipeline/block";
import { Mesh } from "./pipeline/mesh";
import { Cull } from "./pipeline/cull";

enum PipelineMode {
	Post,
	Block,
}

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
export const renderMode = gpu.renderMode;
export const contextUniform = new ContextUniform();
export let pipelineMode = PipelineMode.Block;

console.log("maxDepth:", maxDepth);
console.log("gridSize:", gridSize);

const uniforms = [contextUniform];

document.addEventListener("keydown", (ev) => {
	switch (ev.code) {
		case "KeyP":
			pipelineMode++;
			if (pipelineMode >= Object.keys(PipelineMode).length / 2) {
				pipelineMode = 0;
			}
			console.log(
				"Pipeline mode switched to:",
				pipelineMode === PipelineMode.Post ? "Post" : "Block",
			);
			break;
	}
});

// --- Setup Pipelines ---
const noise = new Noise();
const distanceField = new DistanceField();
const post = new Post();
const block = new Block();
const mesh = new Mesh();
const cull = new Cull();
block.mesh = mesh;
block.cull = cull;
post.noise = noise;
post.distanceField = distanceField;

// --- Timing Display ---
const timingDiv = document.createElement("div");
document.body.appendChild(timingDiv);
timingDiv.style.cssText = `
	position: fixed;
	top: 10px;
	left: 10px;
	background: rgba(0, 0, 0, 0.8);
	color: white;
	padding: 10px;
	font-family: monospace;
	font-size: 12px;
	border-radius: 4px;
	pointer-events: none;
	z-index: 1000;
`;

// --- Frame Graph ---
const frameGraph = new FrameGraph();
const frameGraphContainer = document.createElement("div");
frameGraphContainer.style.cssText = `
	position: fixed;
	top: 10px;
	right: 10px;
	z-index: 1000;
	pointer-events: auto;
`;
frameGraphContainer.appendChild(frameGraph.getElement());
document.body.appendChild(frameGraphContainer);

async function runOneTimeSetup() {
	gpu.update();

	for (const uniform of uniforms) {
		uniform.update();
	}

	console.log("Running one-time octree generation and compaction...");
	const setupEncoder = device.createCommandEncoder();
	noise.update(setupEncoder);

	device.queue.submit([setupEncoder.finish()]);

	// Explicitly wait for the GPU to finish all submitted work.
	await device.queue.onSubmittedWorkDone();

	// Now that we know the GPU is done, read back the data.
	await noise.readback();

	// Initialize distance field after noise pipeline
	await distanceField.init(noise.noiseBuffer, contextUniform.uniformBuffer);
	mesh.init(noise);
	cull.init(noise, mesh);

	// Generate distance field from voxel data
	console.log("Generating distance field...");
	const encoder = device.createCommandEncoder();

	distanceField.update(encoder);
	mesh.update(encoder);
	device.queue.submit([encoder.finish()]);
	await device.queue.onSubmittedWorkDone();

	await mesh.readback();

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

	for (const uniform of uniforms) {
		uniform.update();
	}

	const updateEncoder = device.createCommandEncoder();

	switch (pipelineMode) {
		case PipelineMode.Post:
			post.update(updateEncoder);
			break;
		case PipelineMode.Block:
			// Update culling every frame (async on GPU)
			cull.update(updateEncoder);
			block.update(updateEncoder);
			break;
	}

	device.queue.submit([updateEncoder.finish()]);

	post.afterUpdate();
	block.afterUpdate();

	// Get current renderer's timing based on pipeline mode
	const currentRenderTime =
		pipelineMode === PipelineMode.Post ? post.renderTime : block.renderTime;
	const currentModeName = pipelineMode === PipelineMode.Post ? "Post" : "Block";

	// Update frame graph with GPU render time from active renderer
	frameGraph.addFrameTime(currentRenderTime);
	frameGraph.render();

	// Calculate CPU frame time (excludes GPU work) - moved to capture all CPU work
	const frameEnd = performance.now();
	const cpuFrameTime = frameEnd - frameStart;

	// Update timing display
	const stats = frameGraph.getCurrentStats();
	timingDiv.innerHTML = `
        <b>One-Time Setup</b><br>
		Octree Gen: ${noise.octreeTime.toFixed(3)} ms<br>
        <b>Per Frame (${currentModeName})</b><br>
		GPU Render: ${currentRenderTime.toFixed(3)} ms<br>
		CPU Frame: ${cpuFrameTime.toFixed(3)} ms<br>
		FPS: ${stats ? stats.fps.toFixed(1) : "0.0"}
	`;

	requestAnimationFrame(loop);
}
