import {Chunk} from "./chunk";
import {camera, gridSize} from "../index";

export class Streaming {

	grid = new Map<number, Chunk>();
	generationQueue = new Array<Chunk>();
	generatedChunks = new Array<Chunk>();


	map3D1D(p: number[]): number {
		return p[0] + 16384 * (p[1] + 16384 * p[2]);
	}

	get cameraPositionInGridSpace(): number[] {
		return [
			Math.floor(camera.position[0] / gridSize),
			Math.floor(camera.position[1] / gridSize),
			Math.floor(camera.position[2] / gridSize)
		];
	}

	update() {
		const center = this.cameraPositionInGridSpace;
		const index = this.map3D1D(center);

		if (!this.grid.has(index)) {
			const chunk = new Chunk(index, center);
			this.generationQueue.push(chunk);
			this.grid.set(index, chunk);
		}


	}

}