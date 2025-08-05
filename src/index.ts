import {GPUContext} from "./gpu";
import {Post} from "./pipeline/post";
import {ContextUniform} from "./data/context";
import {Noise} from "./pipeline/noise";

export const gpu = new GPUContext();
await gpu.init();

export const device = gpu.device;
export const context = gpu.context;
export const canvas = gpu.canvas;
export const mouse = gpu.mouse;
export const time = gpu.time;
export const contextUniform = new ContextUniform();


const uniforms = [contextUniform];
const pipelines = [new Noise(), new Post()];

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
		pipeline.afterUpdate(undefined);
	}

	//requestAnimationFrame(loop);
}