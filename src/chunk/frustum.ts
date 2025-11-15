// Frustum plane representation (ax + by + cz + d = 0)
export class FrustumPlane {
	a: number;
	b: number;
	c: number;
	d: number;

	constructor(a: number, b: number, c: number, d: number) {
		// Normalize the plane
		const length = Math.sqrt(a * a + b * b + c * c);
		this.a = a / length;
		this.b = b / length;
		this.c = c / length;
		this.d = d / length;
	}

	// Calculate signed distance from point to plane
	distanceToPoint(x: number, y: number, z: number): number {
		return this.a * x + this.b * y + this.c * z + this.d;
	}
}

export class Frustum {
	planes: FrustumPlane[] = [];

	// Extract frustum planes from view-projection matrix
	// Matrix is in column-major order as used by wgpu-matrix
	extractFromViewProjection(vp: Float32Array) {
		this.planes = [];

		// Left plane: vp[3] + vp[0], vp[7] + vp[4], vp[11] + vp[8], vp[15] + vp[12]
		this.planes.push(new FrustumPlane(
			vp[3] + vp[0],
			vp[7] + vp[4],
			vp[11] + vp[8],
			vp[15] + vp[12]
		));

		// Right plane: vp[3] - vp[0], vp[7] - vp[4], vp[11] - vp[8], vp[15] - vp[12]
		this.planes.push(new FrustumPlane(
			vp[3] - vp[0],
			vp[7] - vp[4],
			vp[11] - vp[8],
			vp[15] - vp[12]
		));

		// Bottom plane: vp[3] + vp[1], vp[7] + vp[5], vp[11] + vp[9], vp[15] + vp[13]
		this.planes.push(new FrustumPlane(
			vp[3] + vp[1],
			vp[7] + vp[5],
			vp[11] + vp[9],
			vp[15] + vp[13]
		));

		// Top plane: vp[3] - vp[1], vp[7] - vp[5], vp[11] - vp[9], vp[15] - vp[13]
		this.planes.push(new FrustumPlane(
			vp[3] - vp[1],
			vp[7] - vp[5],
			vp[11] - vp[9],
			vp[15] - vp[13]
		));

		// Near plane: vp[3] + vp[2], vp[7] + vp[6], vp[11] + vp[10], vp[15] + vp[14]
		this.planes.push(new FrustumPlane(
			vp[3] + vp[2],
			vp[7] + vp[6],
			vp[11] + vp[10],
			vp[15] + vp[14]
		));

		// Far plane: vp[3] - vp[2], vp[7] - vp[6], vp[11] - vp[10], vp[15] - vp[14]
		this.planes.push(new FrustumPlane(
			vp[3] - vp[2],
			vp[7] - vp[6],
			vp[11] - vp[10],
			vp[15] - vp[14]
		));
	}

	// Test if an axis-aligned bounding box intersects with the frustum
	// Returns true if the box is at least partially inside the frustum
	intersectsAABB(minX: number, minY: number, minZ: number, maxX: number, maxY: number, maxZ: number): boolean {
		for (const plane of this.planes) {
			// Find the positive vertex (furthest along plane normal)
			const pX = plane.a >= 0 ? maxX : minX;
			const pY = plane.b >= 0 ? maxY : minY;
			const pZ = plane.c >= 0 ? maxZ : minZ;

			// If the positive vertex is behind the plane, the box is completely outside
			if (plane.distanceToPoint(pX, pY, pZ) < 0) {
				return false;
			}
		}
		return true;
	}
}
