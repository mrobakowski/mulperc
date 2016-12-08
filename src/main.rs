#![feature(iter_max_by)]
#![feature(proc_macro)]
#![feature(conservative_impl_trait)]

extern crate rand;
extern crate rayon;
extern crate nalgebra as na;
extern crate clap;
extern crate pbr;
extern crate nfd;

#[macro_use] extern crate serde_derive;
extern crate serde;
extern crate bincode;

#[macro_use] extern crate conrod;
extern crate image;
extern crate find_folder;

#[macro_use] mod util;
mod multilayer_perceptron;
mod activation_func;
mod img;
mod validators;
mod args;
mod window;
mod window_gui;
mod mnist;
mod gzip;
mod autoencoder;
mod classifier;
mod map_in_place;

fn main() {
    let matches = args::get();
    if let Some(matches) = matches.subcommand_matches("learn") {
        classifier::learn(matches);
    } else if let Some(matches) = matches.subcommand_matches("check") {
        classifier::check(matches);
    } else if let Some(_) = matches.subcommand_matches("gui") {
        window::window_loop();
    } else if let Some(_) = matches.subcommand_matches("autoencoder") {
        autoencoder::run().unwrap();
    } else {
        window::window_loop();
    }
}