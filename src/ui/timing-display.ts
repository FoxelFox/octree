import { UIPanel } from "./ui-panel";

export class TimingDisplay extends UIPanel {
	private timingDiv: HTMLDivElement;

	constructor() {
		super();
		this.timingDiv = this.element as HTMLDivElement;
		this.timingDiv.style.minWidth = '160px';
	}

	protected createElement(): HTMLElement {
		return document.createElement("div");
	}

	public update(
		currentRenderTime: number,
		lightRenderTime: number,
		cullRenderTime: number,
		cpuFrameTime: number,
		stats: {
		fps: number;
		max: number
	} | null,
		meshletCount: number,
	) {
		this.timingDiv.innerHTML = `
			GPU Render: ${currentRenderTime.toFixed(3)} ms<br>
			Light: ${lightRenderTime.toFixed(3)} ms<br>
			Cull: ${cullRenderTime.toFixed(3)} ms<br>
			CPU Frame: ${cpuFrameTime.toFixed(3)} ms<br>
			FPS: ${stats ? stats.fps.toFixed(1) : "0.0"}<br>
			Meshlets: ${meshletCount.toFixed(0)}<br>
    `;
	}
}
