import { UIPanel } from "./ui-panel";

export type LayoutRegion = 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';

export class UIManager {
	private containers: Map<LayoutRegion, HTMLDivElement>;

	constructor() {
		this.containers = new Map();
		this.initializeContainers();
	}

	private initializeContainers(): void {
		const regions: LayoutRegion[] = ['top-left', 'top-right', 'bottom-left', 'bottom-right'];

		for (const region of regions) {
			const container = document.createElement('div');
			container.style.cssText = `
				position: fixed;
				display: flex;
				flex-direction: column;
				gap: 10px;
				z-index: 1000;
				${this.getRegionPositioning(region)}
			`;
			document.body.appendChild(container);
			this.containers.set(region, container);
		}
	}

	private getRegionPositioning(region: LayoutRegion): string {
		switch (region) {
			case 'top-left':
				return 'top: 10px; left: 10px;';
			case 'top-right':
				return 'top: 10px; right: 10px;';
			case 'bottom-left':
				return 'bottom: 10px; left: 10px;';
			case 'bottom-right':
				return 'bottom: 10px; right: 10px;';
		}
	}

	public addPanel(panel: UIPanel, region: LayoutRegion): void {
		const container = this.containers.get(region);
		if (container) {
			container.appendChild(panel.getElement());
		}
	}

	public addElement(element: HTMLElement, region: LayoutRegion): void {
		const container = this.containers.get(region);
		if (container) {
			container.appendChild(element);
		}
	}

	public removePanel(panel: UIPanel): void {
		const element = panel.getElement();
		if (element.parentElement) {
			element.parentElement.removeChild(element);
		}
	}

	public removeElement(element: HTMLElement): void {
		if (element.parentElement) {
			element.parentElement.removeChild(element);
		}
	}
}
