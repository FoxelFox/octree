export abstract class UIPanel {
	protected element: HTMLElement;

	constructor() {
		this.element = this.createElement();
		this.applyBaseStyles();
	}

	protected abstract createElement(): HTMLElement;

	protected applyBaseStyles(): void {
		Object.assign(this.element.style, {
			background: 'rgba(0, 0, 0, 0.8)',
			color: 'white',
			padding: '10px',
			fontFamily: 'monospace',
			fontSize: '12px',
			borderRadius: '4px',
			pointerEvents: 'none',
		});
	}

	public getElement(): HTMLElement {
		return this.element;
	}

	public setPointerEvents(enabled: boolean): void {
		this.element.style.pointerEvents = enabled ? 'auto' : 'none';
	}
}
