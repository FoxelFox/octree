import {canvas, device, mouse, time} from "../index";
import {mat4} from "wgpu-matrix";

export class ContextUniform {
	uniformArray: Float32Array = new Float32Array(
		8   + // stuff
		4*4 + // view
		4*4   // perspective
	);
	uniformBuffer: GPUBuffer;
	canvas = document.getElementsByTagName('canvas')[0];

	constructor() {
		this.uniformBuffer = device.createBuffer({
			size: this.uniformArray.byteLength,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
	}

	update() {
		this.uniformArray[0] = canvas.width;
		this.uniformArray[1] = canvas.height;
		this.uniformArray[2] = mouse.ax;
		this.uniformArray[3] = mouse.ay;
		this.uniformArray[4] = mouse.rx;
		this.uniformArray[5] = mouse.ry;
		this.uniformArray[6] = time.now;
		this.uniformArray[7] = time.delta;

		const eye = [3, 5, 10];
		const target = [2, 2, 2];
		const up = [0, 1, 0];
		const view = mat4.lookAt(eye, target, up);
		this.uniformArray.set(view, 8);

		const fov = 60 * Math.PI / 180;
		const aspect = this.canvas.width / this.canvas.height;
		const near = 0.1
		const far = 1000
		const perspective = mat4.perspective(fov, aspect, near, far);
		this.uniformArray.set(perspective, 8 + 16);

		console.log(perspective)

		device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformArray);
	}
}