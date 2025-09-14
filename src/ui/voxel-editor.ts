import { GPUContext } from "../gpu";
import { VoxelEditor as BaseVoxelEditor } from "../pipeline/generation/voxel_editor";

export class VoxelEditorHandler {
  private voxelEditor: BaseVoxelEditor;
  private isEditingVoxel = false;
  private lastLeftPressed = false;
  private lastRightPressed = false;
  private gpu: GPUContext;

  constructor(gpu: GPUContext, voxelEditor: BaseVoxelEditor) {
    this.gpu = gpu;
    this.voxelEditor = voxelEditor;
  }

  public handleVoxelEditing() {
    // Prevent multiple simultaneous editing operations
    if (this.isEditingVoxel) return;

    const leftPressed = this.gpu.mouse.leftPressed;
    const rightPressed = this.gpu.mouse.rightPressed;

    // Only process on button press (transition from not pressed to pressed)
    const leftJustPressed = leftPressed && !this.lastLeftPressed;
    const rightJustPressed = rightPressed && !this.lastRightPressed;

    // Update last pressed state
    this.lastLeftPressed = leftPressed;
    this.lastRightPressed = rightPressed;

    // Only process if a button was just pressed
    if (!leftJustPressed && !rightJustPressed) return;

    this.isEditingVoxel = true;

    // Read position at screen center (async but non-blocking)
    this.voxelEditor
      .readPositionAtCenter()
      .then((worldPosition) => {
        if (worldPosition && this.voxelEditor.hasGeometryAtCenter()) {
          const editRadius = 10.0; // Configurable brush size

          if (leftJustPressed) {
            // Left click: Add voxels (now non-blocking)
            this.voxelEditor.addVoxels(worldPosition, editRadius);
            console.log(
              "Queued add voxels at:",
              worldPosition[0],
              worldPosition[1],
              worldPosition[2],
            );
          } else if (rightJustPressed) {
            // Right click: Remove voxels (now non-blocking)
            this.voxelEditor.removeVoxels(worldPosition, editRadius);
            console.log(
              "Queued remove voxels at:",
              worldPosition[0],
              worldPosition[1],
              worldPosition[2],
            );
          }
        }
        this.isEditingVoxel = false;
      })
      .catch((error) => {
        console.error("Error during voxel editing:", error);
        this.isEditingVoxel = false;
      });
  }
}