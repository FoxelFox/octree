import { canvas, device, mouse, time } from "../index";

export class ContextUniform {
  uniformArray: Float32Array = new Float32Array(8);
  uniformBuffer: GPUBuffer;

  constructor() {
    this.uniformBuffer = device.createBuffer({
      size: this.uniformArray.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
  }

  update() {
    this.uniformArray[0] = canvas.width;
    this.uniformArray[1] = canvas.height;
    this.uniformArray[2] = mouse.ax;
    this.uniformArray[3] = mouse.ay;
    this.uniformArray[4] = mouse.rx;
    this.uniformArray[5] = mouse.ry;
    this.uniformArray[6] = time.now;
    this.uniformArray[7] = time.delta;
  }
}