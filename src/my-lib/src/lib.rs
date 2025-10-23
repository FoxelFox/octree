extern crate wasm_bindgen;

use wasm_bindgen::prelude::wasm_bindgen;
use wasm_bindgen::JsValue;
use js_sys::{Uint32Array, Float32Array};

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
}

#[wasm_bindgen]
pub fn generate_mesh_direct(
    x: i32,
    y: i32,
    z: i32,
    size: u32,
) -> MeshResult {
    let chunk = mesh::generate_mesh(x, y, z, size);

    unsafe {
        // Create JS-owned copies of the data (not views into WASM memory)
        let vertices_slice = std::slice::from_raw_parts(chunk.vertices(), chunk.vertices_len());
        let normals_slice = std::slice::from_raw_parts(chunk.normals(), chunk.normals_len());
        let colors_slice = std::slice::from_raw_parts(chunk.colors(), chunk.colors_len());
        let material_colors_slice = std::slice::from_raw_parts(chunk.material_colors(), chunk.material_colors_len());
        let commands_slice = std::slice::from_raw_parts(
            chunk.commands() as *const u32,
            chunk.commands_len() * 4 // 4 u32s per command
        );
        let densities_slice = std::slice::from_raw_parts(chunk.densities(), chunk.density_len());
        let vertex_counts_slice = std::slice::from_raw_parts(chunk.vertex_counts(), chunk.vertex_counts_len());

        MeshResult {
            vertices: Float32Array::from(vertices_slice),
            normals: Float32Array::from(normals_slice),
            colors: Uint32Array::from(colors_slice),
            material_colors: Uint32Array::from(material_colors_slice),
            commands: Uint32Array::from(commands_slice),
            densities: Uint32Array::from(densities_slice),
            vertex_counts: Uint32Array::from(vertex_counts_slice),
        }
    }
}

// Generate mesh and write directly to SharedArrayBuffer-backed arrays
#[wasm_bindgen]
pub fn generate_mesh_to_shared(
    x: i32,
    y: i32,
    z: i32,
    size: u32,
    vertices_buf: Float32Array,
    normals_buf: Float32Array,
    colors_buf: Uint32Array,
    material_colors_buf: Uint32Array,
    commands_buf: Uint32Array,
    densities_buf: Uint32Array,
    vertex_counts_buf: Uint32Array,
) -> MeshMetadata {
    // Generate the mesh normally
    let chunk = mesh::generate_mesh(x, y, z, size);

    // Copy to the provided buffers using the copy_within approach
    unsafe {
        let src_vertices = std::slice::from_raw_parts(chunk.vertices() as *const u8, chunk.vertices_len() * 4);
        js_sys::Uint8Array::new(&vertices_buf.buffer()).subarray(0, src_vertices.len() as u32).copy_from(src_vertices);

        let src_normals = std::slice::from_raw_parts(chunk.normals() as *const u8, chunk.normals_len() * 4);
        js_sys::Uint8Array::new(&normals_buf.buffer()).subarray(0, src_normals.len() as u32).copy_from(src_normals);

        let src_colors = std::slice::from_raw_parts(chunk.colors() as *const u8, chunk.colors_len() * 4);
        js_sys::Uint8Array::new(&colors_buf.buffer()).subarray(0, src_colors.len() as u32).copy_from(src_colors);

        let src_material_colors = std::slice::from_raw_parts(chunk.material_colors() as *const u8, chunk.material_colors_len() * 4);
        js_sys::Uint8Array::new(&material_colors_buf.buffer()).subarray(0, src_material_colors.len() as u32).copy_from(src_material_colors);

        let src_commands = std::slice::from_raw_parts(chunk.commands() as *const u8, chunk.commands_len() * 4 * 4);
        js_sys::Uint8Array::new(&commands_buf.buffer()).subarray(0, src_commands.len() as u32).copy_from(src_commands);

        let src_densities = std::slice::from_raw_parts(chunk.densities() as *const u8, chunk.density_len() * 4);
        js_sys::Uint8Array::new(&densities_buf.buffer()).subarray(0, src_densities.len() as u32).copy_from(src_densities);

        let src_vertex_counts = std::slice::from_raw_parts(chunk.vertex_counts() as *const u8, chunk.vertex_counts_len() * 4);
        js_sys::Uint8Array::new(&vertex_counts_buf.buffer()).subarray(0, src_vertex_counts.len() as u32).copy_from(src_vertex_counts);
    }

    MeshMetadata {
        vertices_length: chunk.vertices_len(),
        normals_length: chunk.normals_len(),
        colors_length: chunk.colors_len(),
        material_colors_length: chunk.material_colors_len(),
        commands_length: chunk.commands_len() * 4,
        densities_length: chunk.density_len(),
        vertex_counts_length: chunk.vertex_counts_len(),
    }
}
