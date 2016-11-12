use clap::{Arg, App, SubCommand, ArgMatches};
use validators::*;

pub fn get() -> ArgMatches<'static> {
    App::new("Multilayer-perceptron-based Classifier")
        .version("1.0")
        .author("Mikołaj Robakowski <mikolaj.rob@gmail.com>")
        .about("Classifies images using multilayer perceptron.\n\
                Written for Neural Networks classes at Wrocław University of technology")
        .arg(Arg::with_name("learn-dataset")
            .help("Sets the input folder with the images to learn.\n\
                   The filenames must be in format LABEL(_.*)?")
            .index(1)
            .takes_value(true)
            .value_name("LEARN_DIR")
            .required(true)
            .validator(path_exists))
        .arg(Arg::with_name("learn-sample")
            .short("s")
            .long("sample")
            .help("Sets the percentage of the learn dataset that will be used during each epoch.")
            .takes_value(true)
            .default_value("0.2")
            .validator(str_is_float))
        .arg(Arg::with_name("check-dataset")
            .help("Sets the folder with the images to check against after learning.\n\
                   The filenames must be in format LABEL(_.*)?.\n\
                   By default LEARN_DIR is used.")
            .long("check-dataset")
            .short("c")
            .takes_value(true)
            .value_name("DIR")
            .validator(path_exists))
        .arg(Arg::with_name("max-epochs")
            .help("Sets the maximum number of epochs that the learning algorithm will use.")
            .short("e")
            .long("max-epochs")
            .takes_value(true)
            .default_value("50")
            .validator(str_is_integer))
        .arg(Arg::with_name("learning-rate")
            .short("r")
            .long("learning-rate")
            .help("Sets the learning rate.")
            .takes_value(true)
            .default_value("0.1")
            .validator(str_is_float))
        .get_matches()
}