extern crate wasm_bindgen;

use wasm_bindgen::prelude::wasm_bindgen;
use wasm_bindgen::JsValue;

mod mesh;
mod noise;

#[wasm_bindgen]
pub fn get_memory() -> JsValue {
    wasm_bindgen::memory()
}
