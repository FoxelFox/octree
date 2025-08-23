export class FrameGraph {
	private canvas: HTMLCanvasElement;
	private ctx: CanvasRenderingContext2D;
	private frameHistory: number[] = [];
	private maxHistorySize = 600; // 10 seconds at 60fps
	private targetFrameTime = 16.67; // 60 FPS target
	private lastRenderTime = 0;
	private renderThrottleMs = 33; // ~30 FPS update rate
	
	// Visual constants
	private readonly width = 400;
	private readonly height = 120;
	private readonly padding = { top: 10, right: 20, bottom: 20, left: 40 };
	private readonly gridColor = 'rgba(255, 255, 255, 0.1)';
	private readonly lineColor = '#00ff00';
	private readonly targetLineColor = '#ffff00';
	private readonly warningColor = '#ff8800';
	private readonly errorColor = '#ff0000';

	constructor() {
		this.createCanvas();
		this.setupEventListeners();
		this.render();
	}

	private createCanvas(): void {
		this.canvas = document.createElement('canvas');
		this.canvas.width = this.width;
		this.canvas.height = this.height;
		this.canvas.style.cssText = `
			display: block;
			background: rgba(0, 0, 0, 0.8);
			border: 1px solid rgba(255, 255, 255, 0.2);
			border-radius: 4px;
		`;

		const ctx = this.canvas.getContext('2d');
		if (!ctx) {
			throw new Error('Failed to get 2D canvas context');
		}
		this.ctx = ctx;
		this.ctx.imageSmoothingEnabled = false;
	}

	private setupEventListeners(): void {
		window.addEventListener('resize', this.handleResize.bind(this));
		
		// Add mouse hover for detailed values
		this.canvas.addEventListener('mousemove', this.handleMouseMove.bind(this));
		this.canvas.addEventListener('mouseleave', this.handleMouseLeave.bind(this));
	}

	private handleResize(): void {
		// Debounce resize events
		clearTimeout(this.resizeTimeout);
		this.resizeTimeout = setTimeout(() => {
			this.render();
		}, 100);
	}
	private resizeTimeout: number = 0;

	private handleMouseMove(event: MouseEvent): void {
		const rect = this.canvas.getBoundingClientRect();
		const x = event.clientX - rect.left;
		const y = event.clientY - rect.top;
		
		// Calculate which frame the mouse is over
		const graphWidth = this.width - this.padding.left - this.padding.right;
		const frameIndex = Math.floor((x - this.padding.left) / graphWidth * this.frameHistory.length);
		
		if (frameIndex >= 0 && frameIndex < this.frameHistory.length) {
			const frameTime = this.frameHistory[frameIndex];
			this.showTooltip(x, y, frameTime);
		}
	}

	private handleMouseLeave(): void {
		this.hideTooltip();
	}

	private showTooltip(x: number, y: number, frameTime: number): void {
		// Simple tooltip implementation
		this.canvas.title = `Frame Time: ${frameTime.toFixed(2)}ms (${(1000/frameTime).toFixed(1)} FPS)`;
	}

	private hideTooltip(): void {
		this.canvas.title = '';
	}

	public addFrameTime(frameTime: number): void {
		// Validate frame time
		if (!isFinite(frameTime) || frameTime < 0) {
			return;
		}
		
		// Apply minimum threshold to avoid measurement noise
		const minFrameTime = 0.01; // 0.01ms minimum
		const clampedFrameTime = Math.max(frameTime, minFrameTime);
		
		this.frameHistory.push(clampedFrameTime);
		
		// Keep only the last maxHistorySize frames
		if (this.frameHistory.length > this.maxHistorySize) {
			this.frameHistory.shift();
		}
	}

	public render(): void {
		const now = performance.now();
		if (now - this.lastRenderTime < this.renderThrottleMs) {
			return; // Throttle rendering
		}
		this.lastRenderTime = now;

		try {
			this.clearCanvas();
			this.drawGrid();
			this.drawTargetLine();
			this.drawFrameData();
			this.drawLabels();
		} catch (error) {
			console.warn('FrameGraph render error:', error);
		}
	}

	private clearCanvas(): void {
		this.ctx.clearRect(0, 0, this.width, this.height);
	}

	private drawGrid(): void {
		this.ctx.strokeStyle = this.gridColor;
		this.ctx.lineWidth = 1;

		const graphWidth = this.width - this.padding.left - this.padding.right;
		const graphHeight = this.height - this.padding.top - this.padding.bottom;

		// Vertical grid lines (time markers)
		const verticalLines = 8;
		for (let i = 0; i <= verticalLines; i++) {
			const x = this.padding.left + (i / verticalLines) * graphWidth;
			this.ctx.beginPath();
			this.ctx.moveTo(x, this.padding.top);
			this.ctx.lineTo(x, this.padding.top + graphHeight);
			this.ctx.stroke();
		}

		// Horizontal grid lines (frame time markers)
		const horizontalLines = 6;
		for (let i = 0; i <= horizontalLines; i++) {
			const y = this.padding.top + (i / horizontalLines) * graphHeight;
			this.ctx.beginPath();
			this.ctx.moveTo(this.padding.left, y);
			this.ctx.lineTo(this.padding.left + graphWidth, y);
			this.ctx.stroke();
		}
	}

	private drawTargetLine(): void {
		if (this.frameHistory.length === 0) return;

		const maxTime = Math.max(...this.frameHistory, this.targetFrameTime * 1.5);
		const graphHeight = this.height - this.padding.top - this.padding.bottom;
		const y = this.padding.top + graphHeight - (this.targetFrameTime / maxTime) * graphHeight;

		this.ctx.strokeStyle = this.targetLineColor;
		this.ctx.lineWidth = 1;
		this.ctx.setLineDash([5, 5]);
		
		this.ctx.beginPath();
		this.ctx.moveTo(this.padding.left, y);
		this.ctx.lineTo(this.width - this.padding.right, y);
		this.ctx.stroke();
		
		this.ctx.setLineDash([]);
	}

	private drawFrameData(): void {
		if (this.frameHistory.length < 2) return;

		const graphWidth = this.width - this.padding.left - this.padding.right;
		const graphHeight = this.height - this.padding.top - this.padding.bottom;
		
		// Better scaling: use a minimum scale to make small values visible
		const dataMax = Math.max(...this.frameHistory);
		const minScale = 5.0; // Minimum 5ms scale
		const maxTime = Math.max(dataMax, this.targetFrameTime * 1.5, minScale);

		this.ctx.lineWidth = 2;
		this.ctx.beginPath();

		for (let i = 0; i < this.frameHistory.length; i++) {
			const x = this.padding.left + (i / (this.frameHistory.length - 1)) * graphWidth;
			const frameTime = this.frameHistory[i];
			const y = this.padding.top + graphHeight - (frameTime / maxTime) * graphHeight;

			// Choose color based on performance
			if (frameTime <= this.targetFrameTime) {
				this.ctx.strokeStyle = this.lineColor; // Good performance - green
			} else if (frameTime <= this.targetFrameTime * 1.5) {
				this.ctx.strokeStyle = this.warningColor; // Warning - orange
			} else {
				this.ctx.strokeStyle = this.errorColor; // Poor performance - red
			}

			if (i === 0) {
				this.ctx.moveTo(x, y);
			} else {
				this.ctx.lineTo(x, y);
			}
		}

		this.ctx.stroke();
	}

	private drawLabels(): void {
		this.ctx.fillStyle = 'white';
		this.ctx.font = '10px monospace';
		this.ctx.textAlign = 'left';

		// Y-axis labels (frame times)
		if (this.frameHistory.length > 0) {
			const dataMax = Math.max(...this.frameHistory);
			const minScale = 5.0;
			const maxTime = Math.max(dataMax, this.targetFrameTime * 1.5, minScale);
			const graphHeight = this.height - this.padding.top - this.padding.bottom;
			
			// Bottom label (0ms)
			this.ctx.fillText('0ms', 5, this.padding.top + graphHeight + 15);
			
			// Top label (max time)
			this.ctx.fillText(`${maxTime.toFixed(1)}ms`, 5, this.padding.top + 10);
			
			// Target line label
			const targetY = this.padding.top + graphHeight - (this.targetFrameTime / maxTime) * graphHeight;
			this.ctx.fillText('60fps', this.width - 35, targetY - 5);
		}

	}

	public getElement(): HTMLCanvasElement {
		return this.canvas;
	}

	public getCurrentStats(): { current: number; average: number; max: number; fps: number } | null {
		if (this.frameHistory.length === 0) return null;
		
		const current = this.frameHistory[this.frameHistory.length - 1];
		const average = this.frameHistory.reduce((a, b) => a + b, 0) / this.frameHistory.length;
		const max = Math.max(...this.frameHistory);
		
		// Calculate FPS, handle edge cases
		let fps = 0;
		if (current > 0.001) { // Only calculate if frame time > 1μs
			fps = 1000 / current;
			if (!isFinite(fps) || fps > 10000) {
				fps = 0; // Cap unrealistic FPS values
			}
		}
		
		return { current, average, max, fps };
	}
}