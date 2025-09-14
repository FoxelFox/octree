export class TimingDisplay {
  private timingDiv: HTMLDivElement;

  constructor() {
    this.timingDiv = document.createElement("div");
    document.body.appendChild(this.timingDiv);
    this.timingDiv.style.cssText = `
      position: fixed;
      top: 10px;
      left: 10px;
      background: rgba(0, 0, 0, 0.8);
      color: white;
      padding: 10px;
      font-family: monospace;
      font-size: 12px;
      border-radius: 4px;
      pointer-events: none;
      z-index: 1000;
    `;
  }

  public update(currentRenderTime: number, lightRenderTime: number, cpuFrameTime: number, stats: { fps: number; max: number } | null, meshletCount: number) {
    this.timingDiv.innerHTML = `
      GPU Render: ${currentRenderTime.toFixed(3)} ms<br>
      Light: ${lightRenderTime.toFixed(3)} ms<br>
      CPU Frame: ${cpuFrameTime.toFixed(3)} ms<br>
      FPS: ${stats ? stats.fps.toFixed(1) : "0.0"}<br>
      Meshlets: ${meshletCount.toFixed(0)}<br>
    `;
  }
}