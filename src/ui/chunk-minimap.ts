import {UIPanel} from "./ui-panel";
import {Streaming} from "../chunk/streaming";

export type ChunkStatus = 'queued' | 'processing' | 'finished';

interface ChunkInfo {
	position: [number, number, number];
	lod: number;
	status: ChunkStatus;
}

export class ChunkMinimap extends UIPanel {
	private streaming: Streaming;
	private gridContainer: HTMLDivElement;
	private viewRadius: number = 5; // How many chunks to show in each direction
	private cellSize: number = 10; // Size of each cell in pixels
	private lastCameraChunk: [number, number, number] = [0, 0, 0];
	private cellCache: Map<string, HTMLDivElement> = new Map();
	private lastChunkStates: Map<string, string> = new Map();

	constructor(streaming: Streaming) {
		super();
		this.streaming = streaming;
		this.gridContainer = this.createGridContainer();
		this.element.appendChild(this.gridContainer);

		// Initialize grid on first frame
		const cameraChunkPos = this.streaming.cameraPositionInGridSpace;
		this.lastCameraChunk = [cameraChunkPos[0], cameraChunkPos[1], cameraChunkPos[2]];
		this.rebuildGrid(this.lastCameraChunk);
	}

	public update(): void {
		const cameraChunkPos = this.streaming.cameraPositionInGridSpace;
		const cameraChunk: [number, number, number] = [cameraChunkPos[0], cameraChunkPos[1], cameraChunkPos[2]];

		// Check if camera moved to a different chunk - need full rebuild
		const cameraMoved = this.lastCameraChunk[0] !== cameraChunk[0] ||
		                    this.lastCameraChunk[2] !== cameraChunk[2];

		if (cameraMoved) {
			// Full rebuild when camera moves
			this.rebuildGrid(cameraChunk);
		} else {
			// Incremental update - only update changed cells
			this.updateChangedCells(cameraChunk);
		}

		this.lastCameraChunk = cameraChunk;
	}

	private rebuildGrid(cameraChunk: [number, number, number]): void {
		this.gridContainer.innerHTML = '';
		this.cellCache.clear();
		this.lastChunkStates.clear();

		for (let z = -this.viewRadius; z <= this.viewRadius; z++) {
			for (let x = -this.viewRadius; x <= this.viewRadius; x++) {
				const chunkPos: [number, number, number] = [
					cameraChunk[0] + x,
					0,
					cameraChunk[2] + z
				];

				const isCameraChunk = x === 0 && z === 0;
				const key = this.getPositionKey(x, z);
				const info = this.getChunkStatus(chunkPos);
				const cell = this.createChunkCell(info, isCameraChunk);

				this.gridContainer.appendChild(cell);
				this.cellCache.set(key, cell);
				this.lastChunkStates.set(key, this.getChunkStateHash(info, isCameraChunk));
			}
		}
	}

	private updateChangedCells(cameraChunk: [number, number, number]): void {
		for (let z = -this.viewRadius; z <= this.viewRadius; z++) {
			for (let x = -this.viewRadius; x <= this.viewRadius; x++) {
				const chunkPos: [number, number, number] = [
					cameraChunk[0] + x,
					0,
					cameraChunk[2] + z
				];

				const isCameraChunk = x === 0 && z === 0;
				const key = this.getPositionKey(x, z);
				const info = this.getChunkStatus(chunkPos);
				const stateHash = this.getChunkStateHash(info, isCameraChunk);

				// Only update if state changed
				if (this.lastChunkStates.get(key) !== stateHash) {
					const cell = this.cellCache.get(key);
					if (cell) {
						this.updateChunkCell(cell, info, isCameraChunk);
						this.lastChunkStates.set(key, stateHash);
					}
				}
			}
		}
	}

	private getPositionKey(x: number, z: number): string {
		return `${x},${z}`;
	}

	private getChunkStateHash(info: ChunkInfo | null, isCameraChunk: boolean): string {
		return `${info?.status || 'none'}_${info?.lod || -1}_${isCameraChunk}`;
	}

	protected createElement(): HTMLElement {
		const container = document.createElement('div');
		return container;
	}

	protected applyBaseStyles(): void {
		Object.assign(this.element.style, {
			background: 'rgba(0, 0, 0, 0.9)',
			color: 'white',
			padding: '15px',
			fontFamily: 'monospace',
			borderRadius: '4px',
			pointerEvents: 'none',
		});
	}

	private createGridContainer(): HTMLDivElement {
		const container = document.createElement('div');
		Object.assign(container.style, {
			display: 'grid',
			gap: '2px',
			gridTemplateColumns: `repeat(${this.viewRadius * 2 + 1}, ${this.cellSize}px)`,
			gridTemplateRows: `repeat(${this.viewRadius * 2 + 1}, ${this.cellSize}px)`,
		});
		return container;
	}

	private getChunkStatus(position: [number, number, number]): ChunkInfo | null {
		const chunkIndex = this.streaming.map3D1D(position);

		// Check if chunk is finished
		const chunk = this.streaming.getChunkAt(position);
		if (chunk) {
			return {
				position,
				lod: chunk.lod,
				status: 'finished'
			};
		}

		// Check if chunk is processing (in noise generation or pending generations)
		if (this.streaming.activeNoiseGenerations.has(chunkIndex) ||
			this.streaming.inProgressGenerations.has(chunkIndex)) {
			// Try to find LOD from pending generations
			const pendingTask = this.streaming.pendingGenerations.find(t => t.index === chunkIndex);
			return {
				position,
				lod: pendingTask?.chunk.lod ?? 0,
				status: 'processing'
			};
		}

		return null;
	}

	private getStatusColor(status: ChunkStatus): string {
		switch (status) {
			case 'processing':
				return 'rgba(255, 200, 0, 0.7)'; // Orange - processing
			case 'finished':
				return 'rgba(0, 255, 100, 0.6)'; // Green - finished
		}
	}

	private createChunkCell(info: ChunkInfo | null, isCameraChunk: boolean): HTMLDivElement {
		const cell = document.createElement('div');

		// Set static styles once
		cell.style.width = `${this.cellSize}px`;
		cell.style.height = `${this.cellSize}px`;
		cell.style.display = 'flex';
		cell.style.alignItems = 'center';
		cell.style.justifyContent = 'center';
		cell.style.fontSize = '10px';
		cell.style.borderRadius = '2px';

		// Set dynamic styles
		this.updateChunkCell(cell, info, isCameraChunk);

		return cell;
	}

	private updateChunkCell(cell: HTMLDivElement, info: ChunkInfo | null, isCameraChunk: boolean): void {
		cell.style.border = isCameraChunk ? '2px solid white' : '1px solid rgba(255, 255, 255, 0.2)';
		cell.style.background = info ? this.getStatusColor(info.status) : 'rgba(50, 50, 50, 0.3)';
		cell.textContent = info ? info.lod.toString() : '';
	}
}
