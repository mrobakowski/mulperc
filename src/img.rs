use std::path::Path;
use image;

pub fn get_pixels<P: AsRef<Path>>(p: P) -> Vec<f64> {
    image::open(p).unwrap().to_luma().into_raw().into_iter()
        .map(|byte| 1.0 - (byte as f64 / u8::max_value() as f64)).collect()
}