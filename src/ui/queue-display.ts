import {UIPanel} from "./ui-panel";

export class QueueDisplay extends UIPanel {
	private queueDiv: HTMLDivElement;

	constructor() {
		super();
		this.queueDiv = this.element as HTMLDivElement;
		this.queueDiv.style.minWidth = '160px';
	}

	public update(
		schedulerQueueSize: number,
		schedulerActiveTasks: number,
		schedulerIdleWorkers: number,
		streamingGenerationQueue: number,
		streamingPendingGenerations: number,
		streamingActiveChunks: number,
		streamingTotalChunks: number,
	) {
		this.queueDiv.innerHTML = `
			<strong>Scheduler</strong><br>
			Queue: ${schedulerQueueSize}<br>
			Active: ${schedulerActiveTasks}<br>
			Idle: ${schedulerIdleWorkers}<br>
			<br>
			<strong>Streaming</strong><br>
			Gen Queue: ${streamingGenerationQueue}<br>
			Pending: ${streamingPendingGenerations}<br>
			Active: ${streamingActiveChunks}<br>
			Total: ${streamingTotalChunks}<br>
    `;
	}

	protected createElement(): HTMLElement {
		return document.createElement("div");
	}
}
