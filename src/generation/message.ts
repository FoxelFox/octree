export interface Request {
    id: number
    operation: string
    args: any[]
}

export interface Result {
    id: number
    vertices: Float32Array
    normals: Float32Array
    colors: Uint32Array
    material_colors: Uint32Array
    commands: Uint32Array
    densities: Uint32Array
    vertex_counts: Uint32Array
}