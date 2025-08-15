import {GPUContext} from "./gpu";
import {Post} from "./pipeline/post";
import {ContextUniform} from "./data/context";
import {Noise} from "./pipeline/noise";
import {Compact} from "./pipeline/compact";

export const gpu = new GPUContext();
await gpu.init();

export const maxDepth = 8;
export const gridSize = Math.pow(2, maxDepth);
export const worstCaseMaxNodes = Math.floor((Math.pow(8, maxDepth + 1) -1)/ 7);
export const device = gpu.device;
export const context = gpu.context;
export const canvas = gpu.canvas;
export const mouse = gpu.mouse;
export const camera = gpu.camera;
export const time = gpu.time;
export const renderMode = gpu.renderMode;
export const contextUniform = new ContextUniform();

console.log('maxDepth:', maxDepth);
console.log('gridSize:', gridSize);

const uniforms = [contextUniform];

// --- Setup Pipelines ---
const noise = new Noise();
const compact = new Compact(noise);
const post = new Post();
post.compact = compact;

// --- Timing Display ---
const timingDiv = document.createElement('div');
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

async function runOneTimeSetup() {
	gpu.update();

	for (const uniform of uniforms) {
		uniform.update();
	}

    console.log("Running one-time octree generation and compaction...");
    const setupEncoder = device.createCommandEncoder();
    noise.update(setupEncoder);
    compact.update(setupEncoder);
    device.queue.submit([setupEncoder.finish()]);

    // Explicitly wait for the GPU to finish all submitted work.
    await device.queue.onSubmittedWorkDone();

    // Now that we know the GPU is done, read back the data.
    await noise.readback();
    await compact.readback();
    console.log("Setup complete.");
}

await runOneTimeSetup();
loop();

// this has to be set after first render loop due to safari bug
document.getElementsByTagName('canvas')[0].setAttribute('style', 'position: fixed;')

function loop() {
	gpu.update();

	for (const uniform of uniforms) {
		uniform.update();
	}

	const updateEncoder = device.createCommandEncoder();
	post.update(updateEncoder);
	device.queue.submit([updateEncoder.finish()]);

	post.afterUpdate();

	// Update timing display
	timingDiv.innerHTML = `
        <b>One-Time Setup</b><br>
		Octree Gen: ${noise.octreeTime.toFixed(3)} ms<br>
		Octree Compact: ${compact.compactTime.toFixed(3)} ms<br><br>
        <b>Per Frame</b><br>
		Render: ${post.renderTime.toFixed(3)} ms
	`;

	requestAnimationFrame(loop);
}