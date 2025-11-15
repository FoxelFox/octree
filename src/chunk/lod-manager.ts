import {camera, gridSize} from '../index';

export class LODManager {
	// Distance thresholds in chunk units for switching between LOD levels
	// LOD 0: distance <= LOD_THRESHOLDS[0] (highest quality)
	// LOD 1: distance <= LOD_THRESHOLDS[1]
	// LOD 2: distance > LOD_THRESHOLDS[1] (lowest quality)
	private readonly LOD_THRESHOLDS = [0.5, 2.5];
	private readonly LOD_HYSTERESIS = 0.5;

	/**
	 * Calculate the chunk's distance from the camera in chunk units
	 */
	calculateChunkDistance(position: number[]): number {
		const chunkCenterX = (position[0] + 0.5) * gridSize;
		const chunkCenterY = (position[1] + 0.5) * gridSize;
		const chunkCenterZ = (position[2] + 0.5) * gridSize;

		const dx = camera.position[0] - chunkCenterX;
		const dy = camera.position[1] - chunkCenterY;
		const dz = camera.position[2] - chunkCenterZ;
		return Math.sqrt(dx * dx + dy * dy + dz * dz) / gridSize;
	}

	/**
	 * Determine the appropriate LOD level for a given distance
	 */
	calculateLODForDistance(distance: number): number {
		// LOD levels: 0 = full resolution, 1 = half, 2 = quarter
		for (let lod = 0; lod < this.LOD_THRESHOLDS.length; lod++) {
			if (distance <= this.LOD_THRESHOLDS[lod]) {
				return lod;
			}
		}
		return this.LOD_THRESHOLDS.length;
	}

	/**
	 * Check if LOD should be updated based on current LOD and distance with hysteresis
	 */
	shouldUpdateLOD(currentLOD: number, distance: number): boolean {
		// Calculate target LOD with hysteresis to prevent rapid switching
		// Hysteresis makes it harder to change LOD - we need to be clearly past the threshold
		let targetLOD = 0;
		for (let lod = 0; lod < this.LOD_THRESHOLDS.length; lod++) {
			const threshold = this.LOD_THRESHOLDS[lod];

			// Apply hysteresis based on where we currently are
			// If we're at a higher LOD (lod+1 or more), need to get clearly below threshold to upgrade
			// If we're at a lower LOD (lod or less), need to get clearly above threshold to downgrade
			let adjustedThreshold: number;
			if (currentLOD > lod) {
				// Currently at higher LOD than lod, need distance < threshold - hysteresis to upgrade
				adjustedThreshold = threshold - this.LOD_HYSTERESIS;
			} else {
				// Currently at lower LOD than lod+1, need distance > threshold + hysteresis to downgrade
				adjustedThreshold = threshold + this.LOD_HYSTERESIS;
			}

			if (distance > adjustedThreshold) {
				targetLOD = lod + 1;
			} else {
				break;
			}
		}

		return targetLOD !== currentLOD;
	}

	/**
	 * Check if distance is within range for LOD upgrade (with hysteresis buffer)
	 */
	shouldGenerateBetterLOD(currentLOD: number, distance: number): number | null {
		if (currentLOD === 2 && distance <= this.LOD_THRESHOLDS[1] + this.LOD_HYSTERESIS) {
			return 1;
		} else if (currentLOD === 1 && distance <= this.LOD_THRESHOLDS[0] + this.LOD_HYSTERESIS) {
			return 0;
		}
		return null;
	}

	/**
	 * Check if distance justifies using a better LOD that's already available
	 */
	canUpgradeToLOD(targetLOD: number, distance: number): boolean {
		const threshold = this.LOD_THRESHOLDS[targetLOD];
		const adjustedThreshold = threshold + this.LOD_HYSTERESIS;
		return distance <= adjustedThreshold;
	}
}
