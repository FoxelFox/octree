import { device } from "..";
import { Mesh } from "./mesh";
import { Noise } from "./noise";

export class Cull {
	counter: GPUBuffer;

	init(noise: Noise, mesh: Mesh) {
		this.counter = device.createBuffer({
			size: 4,
			usage:
				GPUBufferUsage.STORAGE |
				GPUBufferUsage.COPY_SRC |
				GPUBufferUsage.COPY_DST,
		});

		device.queue.writeBuffer(this.counter, 0, new Uint32Array([0]));
	}

	update(encoder: GPUCommandEncoder) {}
}
