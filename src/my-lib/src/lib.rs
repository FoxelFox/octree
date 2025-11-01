extern crate wasm_bindgen;

use js_sys::{Float32Array, Uint32Array};
use wasm_bindgen::prelude::wasm_bindgen;
use wasm_bindgen::JsValue;

mod mesh;
mod noise;

#[wasm_bindgen]
pub fn get_memory() -> JsValue {
    wasm_bindgen::memory()
}

#[wasm_bindgen]
pub struct MeshMetadata {
    pub vertices_length: usize,
    pub normals_length: usize,
    pub colors_length: usize,
    pub material_colors_length: usize,
    pub commands_length: usize,
    pub densities_length: usize,
    pub vertex_counts_length: usize,
    pub indices_length: usize,
}

#[wasm_bindgen]
pub struct BufferSizes {
    pub vertices_bytes: usize,
    pub normals_bytes: usize,
    pub colors_bytes: usize,
    pub material_colors_bytes: usize,
    pub commands_bytes: usize,
    pub densities_bytes: usize,
    pub vertex_counts_bytes: usize,
    pub indices_bytes: usize,
}

// Generate mesh and return typed arrays directly (wasm-bindgen handles efficient transfer)
#[wasm_bindgen]
pub struct MeshResult {
    vertices: Float32Array,
    normals: Float32Array,
    colors: Uint32Array,
    material_colors: Uint32Array,
    commands: Uint32Array,
    densities: Uint32Array,
    vertex_counts: Uint32Array,
    indices: Uint32Array,
}

#[wasm_bindgen]
impl MeshResult {
    #[wasm_bindgen(getter)]
    pub fn vertices(&self) -> Float32Array {
        self.vertices.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn normals(&self) -> Float32Array {
        self.normals.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn colors(&self) -> Uint32Array {
        self.colors.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn material_colors(&self) -> Uint32Array {
        self.material_colors.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn commands(&self) -> Uint32Array {
        self.commands.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn densities(&self) -> Uint32Array {
        self.densities.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn vertex_counts(&self) -> Uint32Array {
        self.vertex_counts.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn indices(&self) -> Uint32Array {
        self.indices.clone()
    }
}

#[wasm_bindgen]
pub fn generate_mesh(x: i32, y: i32, z: i32, lod: u32) -> MeshResult {
    let resolution = 256 / (lod + 1);
    let scale = 2_u32.pow(lod) as f32;

    let chunk = mesh::generate_mesh(x, y, z, resolution, scale);

    unsafe {
        // Create JS-owned copies of the data (not views into WASM memory)
        let vertices_slice = std::slice::from_raw_parts(chunk.vertices(), chunk.vertices_len());
        let normals_slice = std::slice::from_raw_parts(chunk.normals(), chunk.normals_len());
        let colors_slice = std::slice::from_raw_parts(chunk.colors(), chunk.colors_len());
        let material_colors_slice =
            std::slice::from_raw_parts(chunk.material_colors(), chunk.material_colors_len());
        let commands_slice = std::slice::from_raw_parts(
            chunk.commands() as *const u32,
            chunk.commands_len() * 5, // 5 u32s per command (DrawIndexedIndirect)
        );
        let densities_slice = std::slice::from_raw_parts(chunk.densities(), chunk.density_len());
        let vertex_counts_slice =
            std::slice::from_raw_parts(chunk.vertex_counts(), chunk.vertex_counts_len());
        let indices_slice = std::slice::from_raw_parts(chunk.indices(), chunk.indices_len());

        MeshResult {
            vertices: Float32Array::from(vertices_slice),
            normals: Float32Array::from(normals_slice),
            colors: Uint32Array::from(colors_slice),
            material_colors: Uint32Array::from(material_colors_slice),
            commands: Uint32Array::from(commands_slice),
            densities: Uint32Array::from(densities_slice),
            vertex_counts: Uint32Array::from(vertex_counts_slice),
            indices: Uint32Array::from(indices_slice),
        }
    }
}
