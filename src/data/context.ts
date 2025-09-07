import {camera, canvas, device, gridSize, maxDepth, mouse, time,} from "../index";
import {mat4} from "wgpu-matrix";

export class ContextUniform {
	uniformArray: Float32Array = new Float32Array(
		10 + // stuff
		2 + // padding
		4 * 4 + // view
		4 * 4 + // inverse view
		4 * 4 + // perspective
		4 * 4 + // inverse perspective
		4 * 4 + // prev view projection
		2 + // jitter offset
		3 + // camera velocity
		1 + // frame count
		1 + // render mode
		1 + // random seed
		1 + // sdf_epsilon
		1 + // sdf_max_steps
		1 + // sdf_over_relaxation
		1 + // taa_enabled
		1 + // hybrid_threshold
		3, // padding to reach 432 bytes (108 floats * 4 bytes = 432)
	);
	uniformBuffer: GPUBuffer;
	canvas = document.getElementsByTagName("canvas")[0];

	private prevViewProjection: Float32Array = new Float32Array(16);
	private frameCount: number = 0;
	private prevCameraPosition: Float32Array = new Float32Array(3);

	constructor() {
		this.uniformBuffer = device.createBuffer({
			size: this.uniformArray.byteLength,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
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

		// Generate TAA jitter pattern (Halton sequence)
		const jitterX = (this.halton(this.frameCount + 1, 2) - 0.5) / canvas.width;
		const jitterY = (this.halton(this.frameCount + 1, 3) - 0.5) / canvas.height;

		const eye = camera.position;
		const target = [
			eye[0] + Math.sin(camera.yaw) * Math.cos(camera.pitch),
			eye[1] + Math.sin(camera.pitch),
			eye[2] + Math.cos(camera.yaw) * Math.cos(camera.pitch),
		];
		const up = [0, 1, 0];
		const view = mat4.lookAt(eye, target, up);
		this.uniformArray.set(view, o);
		o += 16;
		this.uniformArray.set(mat4.inverse(view), o);
		o += 16;

		const fov = (60 * Math.PI) / 180;
		const aspect = this.canvas.width / this.canvas.height;
		const near = 0.1;
		const far = 1000;

		// Create jittered projection matrix for TAA
		const perspective = mat4.perspective(fov, aspect, near, far);
		const jitteredPerspective = mat4.clone(perspective);

		this.uniformArray.set(jitteredPerspective, o);
		o += 16;
		this.uniformArray.set(mat4.inverse(jitteredPerspective), o);
		o += 16;

		// Store previous frame view-projection matrix
		this.uniformArray.set(this.prevViewProjection, o);
		o += 16;

		// Store current jitter offset
		this.uniformArray[o++] = jitterX;
		this.uniformArray[o++] = jitterY;

		o++; // padding
		o++; // padding

		// Calculate camera velocity for TAA
		const currentCameraPosition = [eye[0], eye[1], eye[2]];
		let cameraVelocity = [0, 0, 0];
		if (this.frameCount > 0) {
			cameraVelocity[0] =
				(currentCameraPosition[0] - this.prevCameraPosition[0]) / time.delta;
			cameraVelocity[1] =
				(currentCameraPosition[1] - this.prevCameraPosition[1]) / time.delta;
			cameraVelocity[2] =
				(currentCameraPosition[2] - this.prevCameraPosition[2]) / time.delta;
		}

		// Store camera velocity
		this.uniformArray[o++] = cameraVelocity[0];
		this.uniformArray[o++] = cameraVelocity[1];
		this.uniformArray[o++] = cameraVelocity[2];

		// Store frame count
		integer[o++] = this.frameCount;

		// Store random seed for per-frame noise
		this.uniformArray[o++] = Math.random() * 1000.0;

		// Store distance field parameters
		this.uniformArray[o++] = 0.01; // sdf_epsilon (increased for stability)
		integer[o++] = 256; // sdf_max_steps (optimized for performance)
		this.uniformArray[o++] = 1.0; // sdf_over_relaxation (full stepping)

		// Store hybrid threshold
		this.uniformArray[o++] = 1.0; // hybrid_threshold - switch to SDF when node size <= this value (smaller = less SDF usage)

		// Add padding to reach 432 bytes
		o += 1;
		o += 1;

		// Calculate and store current view-projection for next frame (without jitter for motion vectors)
		const currentViewProjection = mat4.multiply(perspective, view);
		this.prevViewProjection.set(currentViewProjection);

		// Store current camera position for next frame
		this.prevCameraPosition.set(currentCameraPosition);

		this.frameCount++;

		device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformArray);
	}

	// Halton sequence for TAA jitter pattern
	private halton(index: number, base: number): number {
		let result = 0;
		let f = 1;
		while (index > 0) {
			f /= base;
			result += f * (index % base);
			index = Math.floor(index / base);
		}
		return result;
	}
}
