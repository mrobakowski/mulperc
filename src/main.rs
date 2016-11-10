#![feature(conservative_impl_trait)]
#![feature(iter_max_by)]
extern crate rand;
extern crate rayon;
extern crate nalgebra as na;
extern crate image;

mod multilayer_perceptron;
mod activation_func;
mod img;

fn main() {
    println!("Hello, world!");
}