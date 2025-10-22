use crate::noise::only_noise_for_chunk;
use wasm_bindgen::prelude::wasm_bindgen;

// Hand-crafted noise lookup table (256 values)
const EDGE_TABLE_DATA: [u32; 256] = [
    0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03,
    0xe09, 0xf00, 0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c, 0x99c, 0x895, 0xb9f,
    0xa96, 0xd9a, 0xc93, 0xf99, 0xe90, 0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30, 0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6,
    0x6af, 0x5a5, 0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0, 0x460, 0x569,
    0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69,
    0xb60, 0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc, 0xdfc, 0xcf5, 0xfff, 0xef6,
    0x9fa, 0x8f3, 0xbf9, 0xaf0, 0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c, 0xe5c,
    0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950, 0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf,
    0x1c5, 0xcc, 0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0, 0x8c0, 0x9c9, 0xac3,
    0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc, 0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c, 0x15c, 0x55, 0x35f, 0x256, 0x55a,
    0x453, 0x759, 0x650, 0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc, 0x2fc, 0x3f5,
    0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0, 0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65,
    0xc6c, 0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663, 0x569, 0x460, 0xca0, 0xda9, 0xea3, 0xfaa,
    0x8a6, 0x9af, 0xaa5, 0xbac, 0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0, 0xd30,
    0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33,
    0x339, 0x230, 0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c, 0x69c, 0x795, 0x49f,
    0x596, 0x29a, 0x393, 0x99, 0x190, 0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0,
];
const TRIANGLE_TABLE_DATA: [i32; 4096] = [
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 8, 3, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
    8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, 0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9, 2, 10, 0,
    2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1,
    -1, -1, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 11, 2, 8, 11, 0, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1, 3, 10, 1, 11, 10, 3, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, 0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1, 3, 9, 0, 3,
    11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1, 9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 3, 0, 7, 3, 4, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1, 1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, 3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, 9, 2, 10, 9, 0, 2,
    8, 4, 7, -1, -1, -1, -1, -1, -1, -1, 2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1, 8, 4,
    7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1,
    -1, -1, -1, -1, 9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, 4, 7, 11, 9, 4, 11, 9,
    11, 2, 9, 2, 1, -1, -1, -1, -1, 3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1, 1,
    11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1, 4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3,
    -1, -1, -1, -1, 4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1, 9, 5, 4, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, 0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, 5, 4, 8, 3, 5, 3, 1, 5, -1,
    -1, -1, -1, -1, -1, -1, 1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, 0, 8, 1,
    2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1, 5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1,
    -1, 2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1, 9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, 0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1, 0, 5, 4, 0, 1, 5,
    2, 3, 11, -1, -1, -1, -1, -1, -1, -1, 2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1, 10,
    3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, 4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10,
    -1, -1, -1, -1, 5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1, 5, 4, 8, 5, 8, 10, 10,
    8, 11, -1, -1, -1, -1, -1, -1, -1, 9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,
    3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1, 0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1,
    -1, -1, -1, 1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9, 7, 8, 9, 5, 7, 10, 1,
    2, -1, -1, -1, -1, -1, -1, -1, 10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1, 8, 0, 2, 8,
    2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1, 2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1,
    -1, 7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, 9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7,
    11, -1, -1, -1, -1, 2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1, 11, 2, 1, 11, 1, 7, 7,
    1, 5, -1, -1, -1, -1, -1, -1, -1, 9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1, 5, 7,
    0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1, 11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0,
    -1, 11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 10, 6, 5, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, 0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,
    0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1,
    -1, -1, -1, -1, 1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 6, 5, 1, 2, 6, 3,
    0, 8, -1, -1, -1, -1, -1, -1, -1, 9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1, 5, 9,
    8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1, 2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, 11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, 0, 1, 9, 2, 3, 11, 5, 10,
    6, -1, -1, -1, -1, -1, -1, -1, 5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1, 6, 3, 11,
    6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1, 0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1,
    -1, -1, 3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1, 6, 5, 9, 6, 9, 11, 11, 9, 8, -1,
    -1, -1, -1, -1, -1, -1, 5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 3, 0, 4,
    7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1, 1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1,
    -1, 10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1, 6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1,
    -1, -1, -1, -1, 1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1, 8, 4, 7, 9, 0, 5, 0, 6, 5,
    0, 2, 6, -1, -1, -1, -1, 7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1, 3, 11, 2, 7, 8, 4,
    10, 6, 5, -1, -1, -1, -1, -1, -1, -1, 5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1, 0,
    1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, 9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5,
    10, 6, -1, 8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1, 5, 1, 11, 5, 11, 6, 1, 0, 11,
    7, 11, 4, 0, 4, 11, -1, 0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1, 6, 5, 9, 6, 9, 11, 4,
    7, 9, 7, 11, 9, -1, -1, -1, -1, 10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4,
    10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, 10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1,
    -1, -1, -1, -1, 8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1, 1, 4, 9, 1, 2, 4, 2, 6, 4,
    -1, -1, -1, -1, -1, -1, -1, 3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1, 0, 2, 4, 4, 2,
    6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1,
    -1, 10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1, 0, 8, 2, 2, 8, 11, 4, 9, 10, 4,
    10, 6, -1, -1, -1, -1, 3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1, 6, 4, 1, 6, 1, 10,
    4, 8, 1, 2, 1, 11, 8, 11, 1, -1, 9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1, 8, 11, 1,
    8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1, 3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1,
    -1, 6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 7, 10, 6, 7, 8, 10, 8, 9, 10,
    -1, -1, -1, -1, -1, -1, -1, 0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1, 10, 6, 7, 1,
    10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1, 10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1,
    -1, 1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1, 2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7,
    3, 9, -1, 7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1, 7, 3, 2, 6, 7, 2, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, 2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1, 2, 0, 7, 2,
    7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1, 1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1, 11,
    2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1, 8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3,
    6, -1, 0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 7, 8, 0, 7, 0, 6, 3, 11, 0,
    11, 6, 0, -1, -1, -1, -1, 7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 7, 6,
    11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, 0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, 1, 9, 8, 3,
    1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, 10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, 1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, 2, 9, 0, 2, 10, 9, 6, 11, 7,
    -1, -1, -1, -1, -1, -1, -1, 6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1, 7, 2, 3, 6,
    2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1,
    -1, -1, 2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1, 1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7,
    6, -1, -1, -1, -1, 10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1, 10, 7, 6, 1, 7, 10,
    1, 8, 7, 1, 0, 8, -1, -1, -1, -1, 0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1, 7, 6,
    10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1, 6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, 3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1, 8, 6, 11, 8, 4, 6, 9,
    0, 1, -1, -1, -1, -1, -1, -1, -1, 9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1, 6, 8, 4,
    6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1, 1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1,
    -1, -1, 4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1, 10, 9, 3, 10, 3, 2, 9, 4, 3, 11,
    3, 6, 4, 6, 3, -1, 8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, 0, 4, 2, 4, 6, 2, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1, 1, 9,
    4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1, 8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1,
    -1, -1, 10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1, 4, 6, 3, 4, 3, 8, 6, 10, 3, 0,
    3, 9, 10, 9, 3, -1, 10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 9, 5, 7, 6,
    11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1,
    -1, 5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, 11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1,
    5, -1, -1, -1, -1, 9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, 6, 11, 7, 1, 2, 10,
    0, 8, 3, 4, 9, 5, -1, -1, -1, -1, 7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1, 3, 4,
    8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1, 7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1,
    -1, 9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1, 3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1,
    -1, -1, -1, 6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1, 9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3,
    7, -1, -1, -1, -1, 1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1, 4, 0, 10, 4, 10, 5, 0, 3,
    10, 6, 10, 7, 3, 7, 10, -1, 7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1, 6, 9, 5, 6,
    11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1, 3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1,
    -1, 0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1, 6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1,
    -1, -1, -1, -1, -1, 1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1, 0, 11, 3, 0, 6, 11,
    0, 9, 6, 5, 6, 9, 1, 2, 10, -1, 11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1, 6, 11, 3,
    6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1, 5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1,
    -1, 9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1, 1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2,
    6, 2, 8, -1, 1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 3, 6, 1, 6, 10, 3, 8,
    6, 5, 6, 9, 8, 9, 6, -1, 10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1, 0, 3, 8, 5, 6,
    10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, 11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 11, 5, 10, 11, 7, 5,
    8, 3, 0, -1, -1, -1, -1, -1, -1, -1, 5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1,
    10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1, 11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1,
    -1, -1, -1, -1, 0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1, 9, 7, 5, 9, 2, 7, 9, 0, 2,
    2, 11, 7, -1, -1, -1, -1, 7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1, 2, 5, 10, 2, 3, 5,
    3, 7, 5, -1, -1, -1, -1, -1, -1, -1, 8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1, 9, 0,
    1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1, 9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2,
    -1, 1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 8, 7, 0, 7, 1, 1, 7, 5, -1,
    -1, -1, -1, -1, -1, -1, 9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1, 9, 8, 7, 5, 9,
    7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1,
    -1, -1, 5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1, 0, 1, 9, 8, 4, 10, 8, 10, 11,
    10, 4, 5, -1, -1, -1, -1, 10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1, 2, 5, 1, 2, 8,
    5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1, 0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1, 0,
    2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1, 9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, 2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1, 5, 10, 2, 5, 2, 4, 4, 2, 0,
    -1, -1, -1, -1, -1, -1, -1, 3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1, 5, 10, 2, 5, 2,
    4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1, 8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1, 0,
    4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1,
    -1, -1, -1, 9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 11, 7, 4, 9, 11, 9,
    10, 11, -1, -1, -1, -1, -1, -1, -1, 0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1, 1,
    10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1, 3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10,
    11, 4, -1, 4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1, 9, 7, 4, 9, 11, 7, 9, 1, 11,
    2, 11, 1, 0, 8, 3, -1, 11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1, 11, 7, 4, 11,
    4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1, 2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1, 9,
    10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1, 3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0,
    10, -1, 1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 9, 1, 4, 1, 7, 7, 1, 3,
    -1, -1, -1, -1, -1, -1, -1, 4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1, 4, 0, 3, 7, 4,
    3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, 9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, 0, 9, 3, 9, 11, 11, 9,
    10, -1, -1, -1, -1, -1, -1, -1, 0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1, 3,
    1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1,
    -1, -1, -1, -1, -1, 3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1, 0, 2, 11, 8, 0, 11,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, 2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1, 9, 10, 2, 0, 9, 2, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, 2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1, 1, 10, 2, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, 0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 3, 8, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1,
];

// Voxel data structure containing density and color
//#[repr(C, packed)]
#[derive(Clone, Copy)]
struct VoxelData {
    density: f32,
    color: u32, // Packed RGBA color
}

// Unpack RGBA color from u32
fn unpack_color(packed_color: u32) -> [f32; 4] {
    let r = (packed_color & 0xFF) as f32 / 255.0;
    let g = ((packed_color >> 8) & 0xFF) as f32 / 255.0;
    let b = ((packed_color >> 16) & 0xFF) as f32 / 255.0;
    let a = ((packed_color >> 24) & 0xFF) as f32 / 255.0;
    [r, g, b, a]
}

// Pack RGBA color to u32
fn pack_color(color: [f32; 4]) -> u32 {
    let r = (color[0].clamp(0.0, 1.0) * 255.0) as u32 & 0xFF;
    let g = (color[1].clamp(0.0, 1.0) * 255.0) as u32 & 0xFF;
    let b = (color[2].clamp(0.0, 1.0) * 255.0) as u32 & 0xFF;
    let a = (color[3].clamp(0.0, 1.0) * 255.0) as u32 & 0xFF;
    (a << 24) | (b << 16) | (g << 8) | r
}

// Interpolate colors for marching cubes edges
fn interpolate_color(color1: u32, color2: u32, val1: f32, val2: f32) -> u32 {
    let isolevel = 0.0;
    if (isolevel - val1).abs() < 0.00001 {
        return color1;
    }
    if (isolevel - val2).abs() < 0.00001 {
        return color2;
    }
    if (val1 - val2).abs() < 0.00001 {
        return color1;
    }

    let mu = (isolevel - val1) / (val2 - val1);
    let c1 = unpack_color(color1);
    let c2 = unpack_color(color2);
    let interpolated = [
        c1[0] + mu * (c2[0] - c1[0]),
        c1[1] + mu * (c2[1] - c1[1]),
        c1[2] + mu * (c2[2] - c1[2]),
        c1[3] + mu * (c2[3] - c1[3]),
    ];
    pack_color(interpolated)
}

fn get_voxel_data(voxels: &[VoxelData], pos: [u32; 3], grid_size: u32) -> VoxelData {
    // Use (grid_size + 1)³ grid size (with border voxels)
    let size = grid_size + 1;
    let index = (pos[2] * size * size + pos[1] * size + pos[0]) as usize;
    voxels[index]
}

fn get_voxel_density(voxels: &[VoxelData], pos: [u32; 3], grid_size: u32) -> f32 {
    get_voxel_data(voxels, pos, grid_size).density
}

fn get_voxel_color(voxels: &[VoxelData], pos: [u32; 3], grid_size: u32) -> u32 {
    get_voxel_data(voxels, pos, grid_size).color
}

fn get_voxel_density_safe(voxels: &[VoxelData], pos: [i32; 3], grid_size: u32) -> f32 {
    // Allow access to (grid_size + 1)³ grid (0 to grid_size) for border voxels
    let size = (grid_size + 1) as i32;
    if pos[0] < 0 || pos[1] < 0 || pos[2] < 0 || pos[0] >= size || pos[1] >= size || pos[2] >= size
    {
        return 1.0; // Outside bounds = solid
    }
    get_voxel_density(
        voxels,
        [pos[0] as u32, pos[1] as u32, pos[2] as u32],
        grid_size,
    )
}

fn get_voxel_color_safe(voxels: &[VoxelData], pos: [i32; 3], grid_size: u32) -> u32 {
    // Allow access to (grid_size + 1)³ grid (0 to grid_size) for border voxels
    let size = (grid_size + 1) as i32;
    if pos[0] < 0 || pos[1] < 0 || pos[2] < 0 || pos[0] >= size || pos[1] >= size || pos[2] >= size
    {
        return 0x808080FF; // Default gray color for outside bounds
    }
    get_voxel_color(
        voxels,
        [pos[0] as u32, pos[1] as u32, pos[2] as u32],
        grid_size,
    )
}

// Cube vertex positions (8 corners of a unit cube)
const CUBE_VERTICES: [[f32; 3]; 8] = [
    [0.0, 0.0, 0.0], // 0
    [1.0, 0.0, 0.0], // 1
    [1.0, 1.0, 0.0], // 2
    [0.0, 1.0, 0.0], // 3
    [0.0, 0.0, 1.0], // 4
    [1.0, 0.0, 1.0], // 5
    [1.0, 1.0, 1.0], // 6
    [0.0, 1.0, 1.0], // 7
];

// Edge connections for interpolation
const EDGE_VERTICES: [[usize; 2]; 12] = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
];

fn interpolate_vertex(p1: [f32; 3], p2: [f32; 3], val1: f32, val2: f32) -> [f32; 3] {
    let isolevel = 0.0;
    if (isolevel - val1).abs() < 0.00001 {
        return p1;
    }
    if (isolevel - val2).abs() < 0.00001 {
        return p2;
    }
    if (val1 - val2).abs() < 0.00001 {
        return p1;
    }

    let mu = (isolevel - val1) / (val2 - val1);
    [
        p1[0] + mu * (p2[0] - p1[0]),
        p1[1] + mu * (p2[1] - p1[1]),
        p1[2] + mu * (p2[2] - p1[2]),
    ]
}

fn interpolate_normal(n1: [f32; 3], n2: [f32; 3], val1: f32, val2: f32) -> [f32; 3] {
    let isolevel = 0.0;

    let normalize = |v: [f32; 3]| -> [f32; 3] {
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if len > 0.0 {
            [v[0] / len, v[1] / len, v[2] / len]
        } else {
            v
        }
    };

    if (isolevel - val1).abs() < 0.00001 {
        return normalize(n1);
    }
    if (isolevel - val2).abs() < 0.00001 {
        return normalize(n2);
    }
    if (val1 - val2).abs() < 0.00001 {
        return normalize(n1);
    }

    let mu = (isolevel - val1) / (val2 - val1);
    normalize([
        n1[0] + mu * (n2[0] - n1[0]),
        n1[1] + mu * (n2[1] - n1[1]),
        n1[2] + mu * (n2[2] - n1[2]),
    ])
}

fn calculate_gradient(voxels: &[VoxelData], pos: [i32; 3], grid_size: u32) -> [f32; 3] {
    let dx = get_voxel_density_safe(voxels, [pos[0] + 1, pos[1], pos[2]], grid_size)
        - get_voxel_density_safe(voxels, [pos[0] - 1, pos[1], pos[2]], grid_size);
    let dy = get_voxel_density_safe(voxels, [pos[0], pos[1] + 1, pos[2]], grid_size)
        - get_voxel_density_safe(voxels, [pos[0], pos[1] - 1, pos[2]], grid_size);
    let dz = get_voxel_density_safe(voxels, [pos[0], pos[1], pos[2] + 1], grid_size)
        - get_voxel_density_safe(voxels, [pos[0], pos[1], pos[2] - 1], grid_size);

    let gradient = [dx, dy, dz];
    let length = (dx * dx + dy * dy + dz * dz).sqrt();
    if length > 0.0001 {
        [
            -gradient[0] / length,
            -gradient[1] / length,
            -gradient[2] / length,
        ]
    } else {
        [0.0, 1.0, 0.0] // Default normal if gradient is zero
    }
}

#[wasm_bindgen]
pub struct Command {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
}

#[wasm_bindgen]
pub struct Chunk {
    densities: Vec<u32>,
    vertex_counts: Vec<u32>,
    commands: Vec<Command>,
    vertices: Vec<f32>,
    normals: Vec<f32>,
    material_colors: Vec<u32>,
    colors: Vec<u32>,
}

#[wasm_bindgen]
impl Chunk {
    pub fn densities(&self) -> *const u32 {
        self.densities.as_ptr()
    }

    pub fn density_len(&self) -> usize {
        self.densities.len()
    }

    pub fn vertex_counts(&self) -> *const u32 {
        self.vertex_counts.as_ptr()
    }

    pub fn vertex_counts_len(&self) -> usize {
        self.vertex_counts.len()
    }

    pub fn vertices(&self) -> *const f32 {
        self.vertices.as_ptr()
    }

    pub fn vertices_len(&self) -> usize {
        self.vertices.len()
    }

    pub fn normals(&self) -> *const f32 {
        self.normals.as_ptr()
    }

    pub fn normals_len(&self) -> usize {
        self.normals.len()
    }

    pub fn material_colors(&self) -> *const u32 {
        self.material_colors.as_ptr()
    }

    pub fn material_colors_len(&self) -> usize {
        self.material_colors.len()
    }

    pub fn colors(&self) -> *const u32 {
        self.colors.as_ptr()
    }

    pub fn colors_len(&self) -> usize {
        self.colors.len()
    }

    pub fn commands(&self) -> *const Command {
        self.commands.as_ptr()
    }

    pub fn commands_len(&self) -> usize {
        self.commands.len()
    }
}

#[wasm_bindgen]
pub fn generate_mesh(x: i32, y: i32, z: i32, size: u32) -> Chunk {
    const COMPRESSION: u32 = 8;
    let s_size = size / COMPRESSION;

    let density_data = only_noise_for_chunk(x, y, z, size);

    // Convert density data to VoxelData with colors
    let voxel_size = size + 1;
    let total_voxels = (voxel_size * voxel_size * voxel_size) as usize;
    let mut voxels = Vec::with_capacity(total_voxels);

    for (idx, density) in density_data.iter().enumerate() {
        // Rainbow colors based on y-axis position (height)
        // Color format: 0xAABBGGRR (little endian RGBA)
        let color = if *density < 0.0 {
            // Calculate y coordinate from flat index (x + y * width + z * width * height)
            let y = (idx / voxel_size as usize) % voxel_size as usize;

            // Create rainbow based on y position (height from 0.0 to 1.0)
            let t = (y as f32 / voxel_size as f32) * 16.0;
            let r = ((t.sin() * 127.5 + 127.5).clamp(0.0, 255.0) as u32) & 0xFF;
            let g = (((t + 2.0).sin() * 127.5 + 127.5).clamp(0.0, 255.0) as u32) & 0xFF;
            let b = (((t + 4.0).sin() * 127.5 + 127.5).clamp(0.0, 255.0) as u32) & 0xFF;
            0xFF000000 | (b << 16) | (g << 8) | r
        } else {
            0xFF808080 // Gray for outside: A=FF, B=80, G=80, R=80
        };
        voxels.push(VoxelData {
            density: *density,
            color,
        });
    }

    let chunk_world_pos = [x * size as i32, y * size as i32, z * size as i32];

    let mut all_vertices = Vec::new();
    let mut all_normals = Vec::new();
    let mut all_colors = Vec::new(); // u32 packed for lit colors (initialized same as material_colors)
    let mut all_material_colors = Vec::new(); // u32 packed for material colors
    let mut commands = Vec::new();
    let mut densities = Vec::new();
    let mut vertex_counts = Vec::new();

    // Process each meshlet (workgroup)
    for gz in 0..s_size {
        for gy in 0..s_size {
            for gx in 0..s_size {
                let actual_id = [gx, gy, gz];

                let mut local_positions = Vec::new();
                let mut local_normals = Vec::new();
                let mut local_colors = Vec::new();
                let mut density = 0;

                // Triple nested loop over COMPRESSION³ cells
                for z in 0..COMPRESSION {
                    for y in 0..COMPRESSION {
                        for x in 0..COMPRESSION {
                            let world_pos = [
                                (x + actual_id[0] * COMPRESSION) as i32,
                                (y + actual_id[1] * COMPRESSION) as i32,
                                (z + actual_id[2] * COMPRESSION) as i32,
                            ];

                            // Get the 8 corner values and colors of the cube
                            let mut cube_values = [0.0f32; 8];
                            let mut cube_colors = [0u32; 8];

                            for i in 0..8 {
                                let corner_offset = CUBE_VERTICES[i];
                                let pos = [
                                    world_pos[0] + corner_offset[0] as i32,
                                    world_pos[1] + corner_offset[1] as i32,
                                    world_pos[2] + corner_offset[2] as i32,
                                ];
                                cube_values[i] = get_voxel_density_safe(&voxels, pos, size);
                                cube_colors[i] = get_voxel_color_safe(&voxels, pos, size);
                            }

                            // Calculate cube configuration index
                            let mut cube_index = 0u32;
                            for i in 0..8 {
                                if cube_values[i] < 0.0 {
                                    cube_index |= 1u32 << i;
                                }
                            }

                            // Skip empty cubes
                            if cube_index == 0 || cube_index == 255 {
                                continue;
                            }

                            density += 1;

                            // Get edge configuration
                            let edges = EDGE_TABLE_DATA[cube_index as usize];
                            if edges == 0 {
                                continue;
                            }

                            // Calculate interpolated vertices, normals, and colors on edges
                            let mut vertex_list = [[0.0f32; 3]; 12];
                            let mut normal_list = [[0.0f32; 3]; 12];
                            let mut color_list = [0u32; 12];

                            // Check each edge bit and interpolate if necessary
                            for i in 0..12 {
                                let edge_bit = 1u32 << i;
                                if (edges & edge_bit) != 0 {
                                    let v1 = EDGE_VERTICES[i][0];
                                    let v2 = EDGE_VERTICES[i][1];

                                    let p1 = [
                                        world_pos[0] as f32 + CUBE_VERTICES[v1][0],
                                        world_pos[1] as f32 + CUBE_VERTICES[v1][1],
                                        world_pos[2] as f32 + CUBE_VERTICES[v1][2],
                                    ];
                                    let p2 = [
                                        world_pos[0] as f32 + CUBE_VERTICES[v2][0],
                                        world_pos[1] as f32 + CUBE_VERTICES[v2][1],
                                        world_pos[2] as f32 + CUBE_VERTICES[v2][2],
                                    ];

                                    vertex_list[i] = interpolate_vertex(
                                        p1,
                                        p2,
                                        cube_values[v1],
                                        cube_values[v2],
                                    );

                                    // Calculate gradients at cube vertices
                                    let corner1 = [
                                        world_pos[0] + CUBE_VERTICES[v1][0] as i32,
                                        world_pos[1] + CUBE_VERTICES[v1][1] as i32,
                                        world_pos[2] + CUBE_VERTICES[v1][2] as i32,
                                    ];
                                    let corner2 = [
                                        world_pos[0] + CUBE_VERTICES[v2][0] as i32,
                                        world_pos[1] + CUBE_VERTICES[v2][1] as i32,
                                        world_pos[2] + CUBE_VERTICES[v2][2] as i32,
                                    ];

                                    let n1 = calculate_gradient(&voxels, corner1, size);
                                    let n2 = calculate_gradient(&voxels, corner2, size);
                                    normal_list[i] = interpolate_normal(
                                        n1,
                                        n2,
                                        cube_values[v1],
                                        cube_values[v2],
                                    );

                                    // Use hard color edges - pick the color from the "inside" vertex (negative density)
                                    if cube_values[v1] < cube_values[v2] {
                                        color_list[i] = cube_colors[v1]; // v1 is more "inside"
                                    } else {
                                        color_list[i] = cube_colors[v2]; // v2 is more "inside"
                                    }
                                }
                            }

                            // Generate triangles using lookup table
                            let base_triangle_index = (cube_index * 16) as usize;

                            let mut tri_idx = 0;
                            while tri_idx < 16 {
                                let idx1 = base_triangle_index + tri_idx;
                                let idx2 = base_triangle_index + tri_idx + 1;
                                let idx3 = base_triangle_index + tri_idx + 2;

                                let edge1 = TRIANGLE_TABLE_DATA[idx1];
                                let edge2 = TRIANGLE_TABLE_DATA[idx2];
                                let edge3 = TRIANGLE_TABLE_DATA[idx3];

                                if edge1 == -1 {
                                    break;
                                }

                                if edge1 >= 0 && edge2 >= 0 && edge3 >= 0 {
                                    let v1 = vertex_list[edge1 as usize];
                                    let v2 = vertex_list[edge2 as usize];
                                    let v3 = vertex_list[edge3 as usize];

                                    let n1 = normal_list[edge1 as usize];
                                    let n2 = normal_list[edge2 as usize];
                                    let n3 = normal_list[edge3 as usize];

                                    let c1 = color_list[edge1 as usize];
                                    let c2 = color_list[edge2 as usize];
                                    let c3 = color_list[edge3 as usize];

                                    // Apply chunk world position offset to vertices
                                    let v1_world = [
                                        v1[0] + chunk_world_pos[0] as f32,
                                        v1[1] + chunk_world_pos[1] as f32,
                                        v1[2] + chunk_world_pos[2] as f32,
                                    ];
                                    let v2_world = [
                                        v2[0] + chunk_world_pos[0] as f32,
                                        v2[1] + chunk_world_pos[1] as f32,
                                        v2[2] + chunk_world_pos[2] as f32,
                                    ];
                                    let v3_world = [
                                        v3[0] + chunk_world_pos[0] as f32,
                                        v3[1] + chunk_world_pos[1] as f32,
                                        v3[2] + chunk_world_pos[2] as f32,
                                    ];

                                    local_positions.push(v1_world);
                                    local_positions.push(v2_world);
                                    local_positions.push(v3_world);

                                    local_normals.push(n1);
                                    local_normals.push(n2);
                                    local_normals.push(n3);

                                    local_colors.push(c1);
                                    local_colors.push(c2);
                                    local_colors.push(c3);
                                }

                                tri_idx += 3;
                            }
                        }
                    }
                }

                let vertex_count = local_positions.len() as u32;
                let first_vertex = (all_vertices.len() / 4) as u32; // Divide by 4 since vertices are vec4<f32>

                // Append local data to global arrays
                all_vertices.extend(local_positions.iter().flat_map(|v| [v[0], v[1], v[2], 1.0]));
                // Normals need padding to 16 bytes (vec3<f32> in storage buffer has 16-byte stride)
                all_normals.extend(local_normals.iter().flat_map(|n| [n[0], n[1], n[2], 0.0]));

                // Store material colors as u32 packed
                all_material_colors.extend(local_colors.iter().copied());

                // Initialize lit colors same as material colors (lighting will update these on GPU)
                all_colors.extend(local_colors.iter().copied());

                vertex_counts.push(vertex_count);
                densities.push(density);

                commands.push(Command {
                    vertex_count,
                    instance_count: 1,
                    first_vertex,
                    first_instance: 0,
                });
            }
        }
    }

    Chunk {
        commands,
        vertex_counts,
        densities,
        vertices: all_vertices,
        normals: all_normals,
        material_colors: all_material_colors,
        colors: all_colors,
    }
}
