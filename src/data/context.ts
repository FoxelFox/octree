import {canvas, device, gridSize, maxDepth, mouse, time} from "../index";
import {mat4, vec3} from "wgpu-matrix";

export class ContextUniform {
	uniformArray: Float32Array = new Float32Array(
		10  + // stuff
		2   + // padding
		4*4 + // view
		4*4 + // inverse view
		4*4 + // perspective
		4*4   // inverse perspective

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
		const integer = new Uint32Array(this.uniformArray.buffer);
		let o = 0;
		this.uniformArray[o++] = canvas.width;
		this.uniformArray[o++] = canvas.height;
		this.uniformArray[o++] = mouse.ax;
		this.uniformArray[o++] = mouse.ay;
		this.uniformArray[o++] = mouse.rx;
		this.uniformArray[o++] = mouse.ry;
		this.uniformArray[o++] = time.now;
		this.uniformArray[o++] = time.delta;
		integer[o++] = gridSize;
		integer[o++] = maxDepth;
		o += 2; // padding

		const target = [gridSize/2, gridSize/2, gridSize/2];
		const eye = [gridSize, gridSize, -gridSize * 2];
		vec3.rotateY(eye, target, time.now, eye);
		const up = [0, 1, 0];
		const view = mat4.lookAt(eye, target, up);
		this.uniformArray.set(view, o);
		o += 16;
		this.uniformArray.set(mat4.inverse(view), o);
		o += 16;

		const fov = 60 * Math.PI / 180;
		const aspect = this.canvas.width / this.canvas.height;
		const near = 0.1
		const far = 1000
		const perspective = mat4.perspective(fov, aspect, near, far);
		this.uniformArray.set(perspective, o);
		o += 16;
		this.uniformArray.set(mat4.inverse(perspective), o);
		o += 16;



		device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformArray);
	}
}