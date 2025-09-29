export class GPUContext {
	device: GPUDevice;
	context: GPUCanvasContext;
	canvas: HTMLCanvasElement;
	mouse: {
		ax: number;
		ay: number;
		rx: number;
		ry: number;
		deltaX: number;
		deltaY: number;
		locked: boolean;
		leftPressed: boolean;
		rightPressed: boolean;
	};
	keys: Set<string>;
	camera: {
		position: [number, number, number];
		yaw: number;
		pitch: number;
		speed: number;
	};
	time: {
		now: number;
		delta: number;
	};
	taaToggle: {
		lastTKeyPressed: boolean;
	};

	constructor() {
		this.canvas = document.getElementsByTagName("canvas")[0];
		this.setCanvasSize();

		this.mouse = {
			ax: this.canvas.width / 2,
			ay: this.canvas.height / 2,
			rx: 0,
			ry: 0,
			deltaX: 0,
			deltaY: 0,
			locked: false,
			leftPressed: false,
			rightPressed: false,
		};
		this.keys = new Set();
		this.camera = {
			position: [32, 240, 0],
			yaw: 0.7,
			pitch: -0.2,
			speed: 16,
		};
		this.time = {now: 0, delta: 0};
		this.taaToggle = {
			lastTKeyPressed: false,
		};

		window.addEventListener("resize", this.setCanvasSize);
		window.addEventListener("mousemove", this.handleMouseMove);
		window.addEventListener("keydown", this.handleKeyDown);
		window.addEventListener("keyup", this.handleKeyUp);
		window.addEventListener("mousedown", this.handleMouseDown);
		window.addEventListener("mouseup", this.handleMouseUp);
		window.addEventListener("contextmenu", this.handleContextMenu);
		this.canvas.addEventListener("click", this.requestPointerLock);
		document.addEventListener(
			"pointerlockchange",
			this.handlePointerLockChange,
		);
	}

	async init() {
		let errorMessage = "";
		try {
			if (!navigator.gpu) {
				errorMessage = "WebGPU not supported in this browser";
			} else {
				const adapter = await navigator.gpu.requestAdapter();
				if (!adapter) {
					errorMessage = "No suitable GPU adapter found";
				} else {
					// Check for f16 support
					if (!adapter.features.has("shader-f16")) {
						errorMessage = "GPU does not support f16 shaders (required for mesh compression)";
					} else {
						try {
							this.device = await adapter.requestDevice({
								requiredFeatures: ["timestamp-query", "indirect-first-instance", "shader-f16"],
								requiredLimits: {
									maxBufferSize: adapter.limits.maxBufferSize,
									maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
									maxColorAttachmentBytesPerSample: 32,
								},
							});
						} catch (deviceError) {
							errorMessage = `Failed to create WebGPU device: ${deviceError.message}`;
						}
					}
				}
			}
		} catch (error) {
			errorMessage = `WebGPU initialization error: ${error.message}`;
		} finally {
			if (!this.device) {
				document.body.innerHTML =
					"" +
					'<div style="padding: 0 16px">' +
					"<h1>No GPU available 😔</h1>" +
					"<p>You need a WebGPU compatible browser</p>" +
					'<a style="color: brown" href="https://caniuse.com/webgpu">https://caniuse.com/webgpu</a>' +
					(errorMessage
						? `<br><br><strong>Error details:</strong><br><code>${errorMessage}</code>`
						: "") +
					"</div> ";
			}
		}
		this.context = this.canvas.getContext("webgpu");
		const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
		this.context.configure({
			device: this.device,
			format: presentationFormat,
			alphaMode: "premultiplied",
		});
	}

	update() {
		const now = performance.now() / 1000;
		this.time.delta = now - this.time.now;
		this.time.now = now;
		this.updateCamera();
	}

	setCanvasSize = () => {
		this.canvas.width = window.innerWidth * devicePixelRatio;
		this.canvas.height = window.innerHeight * devicePixelRatio;
	};

	handleMouseMove = (ev: MouseEvent) => {
		this.mouse.ax = ev.clientX;
		this.mouse.ay = ev.clientY;
		this.mouse.rx =
			((ev.clientX - window.innerWidth / 2) / window.innerWidth) * 2;
		this.mouse.ry =
			((ev.clientY - window.innerHeight / 2) / window.innerHeight) * 2;

		if (this.mouse.locked) {
			this.mouse.deltaX = ev.movementX || 0;
			this.mouse.deltaY = ev.movementY || 0;

			const sensitivity = 0.002;
			this.camera.yaw -= this.mouse.deltaX * sensitivity;
			this.camera.pitch -= this.mouse.deltaY * sensitivity;
			this.camera.pitch = Math.max(
				-Math.PI / 2 + 0.1,
				Math.min(Math.PI / 2 - 0.1, this.camera.pitch),
			);
		}
	};

	handleKeyDown = (ev: KeyboardEvent) => {
		this.keys.add(ev.code.toLowerCase());
	};

	handleKeyUp = (ev: KeyboardEvent) => {
		this.keys.delete(ev.code.toLowerCase());
	};

	handleMouseDown = (ev: MouseEvent) => {
		if (ev.button === 0) { // Left mouse button
			this.mouse.leftPressed = true;
		} else if (ev.button === 2) { // Right mouse button
			this.mouse.rightPressed = true;
			ev.preventDefault(); // Prevent context menu
		}
	};

	handleMouseUp = (ev: MouseEvent) => {
		if (ev.button === 0) { // Left mouse button
			this.mouse.leftPressed = false;
		} else if (ev.button === 2) { // Right mouse button
			this.mouse.rightPressed = false;
		}
	};

	handleContextMenu = (ev: Event) => {
		ev.preventDefault(); // Prevent context menu from appearing
	};

	requestPointerLock = () => {
		this.canvas.requestPointerLock();
	};

	handlePointerLockChange = () => {
		this.mouse.locked = document.pointerLockElement === this.canvas;
	};

	updateCamera() {
		if (!this.mouse.locked) return;

		const speedMultiplier =
			this.keys.has("shiftleft") || this.keys.has("shiftright") ? 3 : 1;
		const moveSpeed = this.camera.speed * this.time.delta * speedMultiplier;

		const forward = [
			Math.sin(this.camera.yaw) * Math.cos(this.camera.pitch),
			Math.sin(this.camera.pitch),
			Math.cos(this.camera.yaw) * Math.cos(this.camera.pitch),
		];

		const right = [Math.cos(this.camera.yaw), 0, -Math.sin(this.camera.yaw)];

		if (this.keys.has("keyw")) {
			this.camera.position[0] += forward[0] * moveSpeed;
			this.camera.position[1] += forward[1] * moveSpeed;
			this.camera.position[2] += forward[2] * moveSpeed;
		}
		if (this.keys.has("keys")) {
			this.camera.position[0] -= forward[0] * moveSpeed;
			this.camera.position[1] -= forward[1] * moveSpeed;
			this.camera.position[2] -= forward[2] * moveSpeed;
		}
		if (this.keys.has("keya")) {
			this.camera.position[0] += right[0] * moveSpeed;
			this.camera.position[2] += right[2] * moveSpeed;
		}
		if (this.keys.has("keyd")) {
			this.camera.position[0] -= right[0] * moveSpeed;
			this.camera.position[2] -= right[2] * moveSpeed;
		}
	}

}
