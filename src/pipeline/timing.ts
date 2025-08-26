import {device} from "../index";

export class RenderTimer {
	private querySet: GPUQuerySet;
	private queryBuffer: GPUBuffer;
	private queryReadbackBuffer: GPUBuffer;
	private isReadingTiming: boolean = false;
	private lastTimingFrame: number = 0;
	private frame: number = 0;
	
	public renderTime: number = 0;

	constructor(private name: string) {
		this.querySet = device.createQuerySet({
			type: 'timestamp',
			count: 2,
		});

		this.queryBuffer = device.createBuffer({
			size: 16,
			usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
		});

		this.queryReadbackBuffer = device.createBuffer({
			size: 16,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
		});
	}

	shouldMeasureTiming(): boolean {
		return !this.isReadingTiming && (this.frame - this.lastTimingFrame) > 0;
	}

	getTimestampWrites(): GPURenderPassTimestampWrites | undefined {
		if (this.shouldMeasureTiming()) {
			return {
				querySet: this.querySet,
				beginningOfPassWriteIndex: 0,
				endOfPassWriteIndex: 1,
			};
		}
		return undefined;
	}

	resolveTimestamps(commandEncoder: GPUCommandEncoder): void {
		if (this.shouldMeasureTiming()) {
			commandEncoder.resolveQuerySet(this.querySet, 0, 2, this.queryBuffer, 0);
			commandEncoder.copyBufferToBuffer(this.queryBuffer, 0, this.queryReadbackBuffer, 0, 16);
			this.lastTimingFrame = this.frame;
		}
		this.frame++;
	}

	readTimestamps(): void {
		if (!this.isReadingTiming && (this.frame - this.lastTimingFrame) === 1) {
			this.isReadingTiming = true;
			this.queryReadbackBuffer.mapAsync(GPUMapMode.READ).then(() => {
				const times = new BigUint64Array(this.queryReadbackBuffer.getMappedRange());
				const startTime = times[0];
				const endTime = times[1];

				if (startTime > 0n && endTime > 0n && endTime >= startTime) {
					const duration = endTime - startTime;
					this.renderTime = Number(duration) / 1_000_000;
				}

				this.queryReadbackBuffer.unmap();
				this.isReadingTiming = false;
			}).catch(() => {
				this.isReadingTiming = false;
			});
		}
	}

	destroy(): void {
		this.querySet.destroy();
		this.queryBuffer.destroy();
		this.queryReadbackBuffer.destroy();
	}
}