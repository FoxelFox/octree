import { FrameGraph } from "./frame-graph";
import { UIPanel } from "./ui-panel";

export class FrameGraphManager extends UIPanel {
  private frameGraph: FrameGraph;

  constructor() {
    super();
    this.frameGraph = new FrameGraph();
    this.element.appendChild(this.frameGraph.getElement());
    this.setPointerEvents(true);
  }

  protected createElement(): HTMLElement {
    const container = document.createElement("div");
    // Remove background from container since FrameGraph has its own
    container.style.background = 'none';
    container.style.padding = '0';
    return container;
  }

  public getFrameGraph() {
    return this.frameGraph;
  }

  public render() {
    this.frameGraph.render();
  }
}