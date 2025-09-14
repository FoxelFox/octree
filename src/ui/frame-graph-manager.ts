import { FrameGraph } from "./frame-graph";

export class FrameGraphManager {
  private frameGraph: FrameGraph;
  private frameGraphContainer: HTMLDivElement;

  constructor() {
    this.frameGraph = new FrameGraph();
    this.frameGraphContainer = document.createElement("div");
    this.frameGraphContainer.style.cssText = `
      position: fixed;
      top: 10px;
      right: 10px;
      z-index: 1000;
      pointer-events: auto;
    `;
    this.frameGraphContainer.appendChild(this.frameGraph.getElement());
    document.body.appendChild(this.frameGraphContainer);
  }

  public getFrameGraph() {
    return this.frameGraph;
  }

  public render() {
    this.frameGraph.render();
  }
}