import {GPUContext} from "./gpu";
import {ContextUniform} from "./data/context";
import {TimingDisplay} from "./ui/timing-display";
import {FrameGraphManager} from "./ui/frame-graph-manager";
import {QueueDisplay} from "./ui/queue-display";
import {UIManager} from "./ui/ui-manager";
import {ChunkMinimap} from "./ui/chunk-minimap";
import {Streaming} from "./chunk/streaming";
import {Scheduler} from "./generation/scheduler";

export const scheduler = new Scheduler();
export const gpu = new GPUContext();
await gpu.init();

// Export all the constants FIRST before creating objects that import them
export const maxDepth = 8;
export const gridSize = Math.pow(2, maxDepth);
export const device = gpu.device;
export const context = gpu.context;
export const canvas = gpu.canvas;
export const mouse = gpu.mouse;
export const camera = gpu.camera;
export const time = gpu.time;
export const compression = 8;

export const contextUniform = new ContextUniform();

// Safari workaround: Wrap initialization in async function to ensure exports are available
async function initializeApp() {
	const uniforms = [contextUniform];

	const streaming = new Streaming();

	// --- UI Components ---
	const uiManager = new UIManager();
	const timingDisplay = new TimingDisplay();
	const frameGraphManager = new FrameGraphManager();
	const queueDisplay = new QueueDisplay();
	const chunkMinimap = new ChunkMinimap(streaming);

	uiManager.addPanel(timingDisplay, 'top-left');
	uiManager.addPanel(queueDisplay, 'top-left');
	uiManager.addPanel(frameGraphManager, 'top-right');
	uiManager.addPanel(chunkMinimap, 'bottom-right');

	function runOneTimeSetup() {
		gpu.update();

		for (const uniform of uniforms) {
			uniform.update();
		}

		streaming.init();
	}

	let renderCount = 0;

	runOneTimeSetup();
	loop();

	function loop() {
		renderCount++;
		let now = performance.now();
		const time = now - lastFrame;
		lastFrame = now;

		gpu.update();

		// Handle voxel editing when the pointer is locked and in Block mode
		if (streaming.voxelEditor && gpu.mouse.locked) {
			streaming.voxelEditorHandler.handleVoxelEditing();
		}

		for (const uniform of uniforms) {
			uniform.update();
		}

		const updateEncoder = device.createCommandEncoder();
		streaming.update(updateEncoder);
		device.queue.submit([updateEncoder.finish()]);
		streaming.afterUpdate().then(() => {
			// Update the frame graph with GPU render time from the active renderer
			frameGraphManager.getFrameGraph().addFrameTime(time);
			frameGraphManager.render();

			// Calculate CPU frame time (excludes GPU work) - moved to capture all CPU work
			const frameEnd = performance.now();
			const cpuFrameTime = frameEnd - now;

			// Update timing display
			const stats = frameGraphManager.getFrameGraph().getCurrentStats();

			timingDisplay.update(
				time,
				cpuFrameTime,
				stats,
				streaming.cull.count,
			);

			queueDisplay.update(
				scheduler.queue.length,
				scheduler.activeTasks.size,
				scheduler.idle.length,
				streaming.generationQueue.length,
				streaming.pendingGenerations.length,
				streaming.activeChunks.size,
				streaming.grid.size,
			);

			chunkMinimap.update();

			if (!gpu.hasError) {
				requestAnimationFrame(loop);
			}
		}).catch(error => {
			console.error("[Main] Error in afterUpdate:", error);
			throw error;
		});
	}
}

// Move these to module scope so they're accessible
let lastFrame = performance.now();

// Start the app
initializeApp();

// this has to be set after first render loop due to safari bug
document
	.getElementsByTagName("canvas")[0]
	.setAttribute("style", "position: fixed;");
