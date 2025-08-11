import {contextUniform, device, gridSize, worstCaseMaxNodes} from "..";
import shader from "./noise.wgsl" with {type: "text"};

export class Noise {

	noiseBuffer: GPUBuffer
	nodeCounterBuffer: GPUBuffer
	nodesBuffer: GPUBuffer
	pipeline: GPUComputePipeline
	bindGroup0: GPUBindGroup
	bindGroup1: GPUBindGroup
	noiseReadbackBuffer: GPUBuffer
	nodesReadbackBuffer: GPUBuffer
	isReading: boolean
	
	// timing
	querySet: GPUQuerySet
	queryBuffer: GPUBuffer
	queryReadbackBuffer: GPUBuffer
	octreeTime: number = 0

	// output
	result: Uint32Array;
	nodesResult: Uint32Array;

	constructor() {
		const size = Math.pow(gridSize, 3) * 4;
		this.noiseBuffer = device.createBuffer({
			label: "Noise",
			size,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
		});

		this.nodeCounterBuffer = device.createBuffer({
			label: "Pointer",
			size: 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});

		this.nodesBuffer = device.createBuffer({
			label: "Octree Nodes",
			size: worstCaseMaxNodes * 9 * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
		})

		this.noiseReadbackBuffer = device.createBuffer({
			size,
			usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
		});

		this.nodesReadbackBuffer = device.createBuffer({
			size: worstCaseMaxNodes * 9 * 4,
			usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
		});

		device.queue.writeBuffer(this.nodeCounterBuffer, 0, new Uint32Array([0]));

		// Create timestamp query set for precise timing
		this.querySet = device.createQuerySet({
			type: 'timestamp',
			count: 2, // start and end timestamps
		});

		this.queryBuffer = device.createBuffer({
			size: 16, // 2 timestamps * 8 bytes each
			usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
		});

		this.queryReadbackBuffer = device.createBuffer({
			size: 16,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
		});

		this.pipeline = device.createComputePipeline({
			label: "Noise",
			layout: "auto",
			compute: {
				module: device.createShaderModule({
					code: shader
				}),
				entryPoint: "main",
			},
		});

		this.bindGroup0 = device.createBindGroup({
			label: "Noise",
			layout: this.pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: this.noiseBuffer},
				{binding: 1, resource: this.nodeCounterBuffer},
				{binding: 2, resource: this.nodesBuffer},
			]
		});

		this.bindGroup1 = device.createBindGroup({
			layout: this.pipeline.getBindGroupLayout(1),
			entries: [{
				binding: 0,
				resource: {buffer: contextUniform.uniformBuffer}
			}]
		});
	}

	update(commandEncoder: GPUCommandEncoder) {


		if(this.result) {
			return;
		}

		if (this.isReading) {
			return;
		}

		console.log('generate noise')
		const computePass = commandEncoder.beginComputePass({
			timestampWrites: {
				querySet: this.querySet,
				beginningOfPassWriteIndex: 0,
				endOfPassWriteIndex: 1,
			},
		});
		computePass.setPipeline(this.pipeline);
		computePass.setBindGroup(0, this.bindGroup0);
		computePass.setBindGroup(1, this.bindGroup1);
		computePass.dispatchWorkgroups(
			Math.ceil(gridSize / 4),
			Math.ceil(gridSize / 4),
			Math.ceil(gridSize / 4)
		);
		computePass.end();

		// Resolve timestamp queries
		commandEncoder.resolveQuerySet(this.querySet, 0, 2, this.queryBuffer, 0);
		commandEncoder.copyBufferToBuffer(this.queryBuffer, 0, this.queryReadbackBuffer, 0, 16);

		const size = Math.pow(gridSize, 3) * 4;


		// read back
		commandEncoder.copyBufferToBuffer(this.noiseBuffer, 0, this.noiseReadbackBuffer, 0, size);
		// TODO this could be optimized by reading the nodes_counter and only copying whats actual needed
		commandEncoder.copyBufferToBuffer(this.nodesBuffer, 0, this.nodesReadbackBuffer, 0, worstCaseMaxNodes * 9 * 4);

	}

	afterUpdate() {

		if (this.result) {
			return;
		}

		if (this.isReading) {
			return;
		}

		this.isReading = true;
		
		// Read timing data
		this.queryReadbackBuffer.mapAsync(GPUMapMode.READ).then(() => {
			const times = new BigUint64Array(this.queryReadbackBuffer.getMappedRange());
			const startTime = times[0];
			const endTime = times[1];
			this.octreeTime = Number(endTime - startTime) / 1_000_000; // Convert to milliseconds
			
			console.log(`Octree generation time: ${this.octreeTime.toFixed(3)} ms`);
			this.queryReadbackBuffer.unmap();
		});
		
		this.noiseReadbackBuffer.mapAsync(GPUMapMode.READ).then(() => {
			const mappedData = new Uint32Array(this.noiseReadbackBuffer.getMappedRange());
			this.result = new Uint32Array(mappedData);

			let ones = 0;
			for (let i = 0; i < this.result.length; i++) {
				if (this.result[i] == 1) {
					ones++;
				}
			}

			console.log('blocks in noise:', ones)

			this.noiseReadbackBuffer.unmap();
		});

		this.nodesReadbackBuffer.mapAsync(GPUMapMode.READ).then(() => {
			const mappedData = new Uint32Array(this.nodesReadbackBuffer.getMappedRange());
			this.nodesResult = new Uint32Array(mappedData);

			let leafs = 0;
			for (let offset = 0; offset < this.nodesResult.length; offset += 9) {
				if (this.nodesResult[offset] == 1) {
					leafs++;
				}
			}
			console.log('leafs in node:', leafs);
			this.nodesReadbackBuffer.unmap();
		});
	}
}