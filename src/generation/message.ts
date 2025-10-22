export interface Request {
    id: number
    operation: string
    args: any[]
    sharedBuffers: {
        vertices: SharedArrayBuffer
        normals: SharedArrayBuffer
        colors: SharedArrayBuffer
        material_colors: SharedArrayBuffer
        commands: SharedArrayBuffer
        densities: SharedArrayBuffer
        vertex_counts: SharedArrayBuffer
    }
}

export interface Result {
    id: number
    metadata: {
        verticesLength: number
        normalsLength: number
        colorsLength: number
        materialColorsLength: number
        commandsLength: number
        densitiesLength: number
        vertexCountsLength: number
    }
}