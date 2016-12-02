use std::path::Path;
use std::fs::File;
use image;

pub fn get_pixels<P: AsRef<Path>>(p: P) -> Vec<f64> {
    image::open(p).unwrap().to_luma().into_raw().into_iter()
        .map(|byte| 1.0 - (byte as f64 / u8::max_value() as f64)).collect()
}

pub fn save<P: AsRef<Path>>(v: &[f64], w: u32, h: u32, p: P) {
//    println!("{:?}", p.as_ref());
    let ref mut fout = File::create(p).unwrap();
    image::DynamicImage::ImageLuma8(image::GrayImage::from_raw(w, h, v.iter()
        .map(|f| ((1.0 - f) * 255.0) as u8).collect()).unwrap()).save(fout, image::PNG).unwrap();
}