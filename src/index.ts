import { GPUContext } from "./gpu";
import { ContextUniform } from "./data/context";
import { Noise } from "./pipeline/noise";
import { FrameGraph } from "./ui/FrameGraph";
import { Block } from "./pipeline/block";
import { Mesh } from "./pipeline/mesh";
import { Cull } from "./pipeline/cull";
import { VoxelEditor } from "./pipeline/voxel_editor";

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
const block = new Block();
const mesh = new Mesh();
const cull = new Cull();
block.mesh = mesh;
block.cull = cull;

// --- Voxel Editor ---
let voxelEditor: VoxelEditor;

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

	mesh.init(noise);
	cull.init(noise, mesh);

	// Generate distance field from voxel data
	console.log("Generating distance field...");
	const encoder = device.createCommandEncoder();

	mesh.update(encoder);
	device.queue.submit([encoder.finish()]);

	mesh.afterUpdate();

	// Initialize voxel editor after all systems are ready
	voxelEditor = new VoxelEditor(block, noise, mesh, cull);

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
		handleVoxelEditing();
	}

	for (const uniform of uniforms) {
		uniform.update();
	}

	const updateEncoder = device.createCommandEncoder();

	// Update culling every frame (async on GPU)
	cull.update(updateEncoder);
	block.update(updateEncoder);

	device.queue.submit([updateEncoder.finish()]);

	block.afterUpdate();
	mesh.afterUpdate();
	cull.afterUpdate();

	// Get current renderer's timing based on pipeline mode
	const currentRenderTime = block.renderTime;

	// Update frame graph with GPU render time from active renderer
	frameGraph.addFrameTime(currentRenderTime);
	frameGraph.render();

	// Calculate CPU frame time (excludes GPU work) - moved to capture all CPU work
	const frameEnd = performance.now();
	const cpuFrameTime = frameEnd - frameStart;

	// Update timing display
	const stats = frameGraph.getCurrentStats();

	timingDiv.innerHTML = `
		GPU Render: ${currentRenderTime.toFixed(3)} ms<br>
		CPU Frame: ${cpuFrameTime.toFixed(3)} ms<br>
		FPS: ${stats ? stats.fps.toFixed(1) : "0.0"}<br>
		Meshlets: ${cull.count}<br>
	`;

	requestAnimationFrame(loop);
}

// Voxel editing logic
let isEditingVoxel = false;
let lastLeftPressed = false;
let lastRightPressed = false;

function handleVoxelEditing() {
	// Prevent multiple simultaneous editing operations
	if (isEditingVoxel) return;

	const leftPressed = gpu.mouse.leftPressed;
	const rightPressed = gpu.mouse.rightPressed;

	// Only process on button press (transition from not pressed to pressed)
	const leftJustPressed = leftPressed && !lastLeftPressed;
	const rightJustPressed = rightPressed && !lastRightPressed;

	// Update last pressed state
	lastLeftPressed = leftPressed;
	lastRightPressed = rightPressed;

	// Only process if a button was just pressed
	if (!leftJustPressed && !rightJustPressed) return;

	isEditingVoxel = true;

	// Read position at screen center (async but non-blocking)
	voxelEditor
		.readPositionAtCenter()
		.then((worldPosition) => {
			if (worldPosition && voxelEditor.hasGeometryAtCenter()) {
				const editRadius = 10.0; // Configurable brush size

				if (leftJustPressed) {
					// Left click: Add voxels (now non-blocking)
					voxelEditor.addVoxels(worldPosition, editRadius);
					console.log(
						"Queued add voxels at:",
						worldPosition[0],
						worldPosition[1],
						worldPosition[2],
					);
				} else if (rightJustPressed) {
					// Right click: Remove voxels (now non-blocking)
					voxelEditor.removeVoxels(worldPosition, editRadius);
					console.log(
						"Queued remove voxels at:",
						worldPosition[0],
						worldPosition[1],
						worldPosition[2],
					);
				}
			}
			isEditingVoxel = false;
		})
		.catch((error) => {
			console.error("Error during voxel editing:", error);
			isEditingVoxel = false;
		});
}
