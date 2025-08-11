import {GPUContext} from "./gpu";
import {Post} from "./pipeline/post";
import {ContextUniform} from "./data/context";
import {Noise} from "./pipeline/noise";
import {Block} from "./pipeline/block";

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
export const contextUniform = new ContextUniform();


console.log('maxDepth:', maxDepth);
console.log('gridSize:', gridSize);

const uniforms = [contextUniform];

const noise = new Noise();
const post = new Post();
//const block = new Block();

//block.noise = noise;
post.noise = noise;

const pipelines = [noise, post];

// Create timing display
const timingDiv = document.createElement('div');
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
document.body.appendChild(timingDiv);

loop();

// this has to be set after first render loop due to safari bug
document.getElementsByTagName('canvas')[0].setAttribute('style', 'position: fixed;')

function loop() {
	gpu.update();

	for (const uniform of uniforms) {
		uniform.update();
	}

	const updateEncoder = device.createCommandEncoder();
	for (const pipeline of pipelines) {
		pipeline.update(updateEncoder);
	}
	device.queue.submit([updateEncoder.finish()]);


	for (const pipeline of pipelines) {
		pipeline.afterUpdate();
	}

	// Update timing display
	timingDiv.innerHTML = `
		Octree: ${noise.octreeTime.toFixed(3)} ms<br>
		Render: ${post.renderTime.toFixed(3)} ms
	`;

	requestAnimationFrame(loop);
}