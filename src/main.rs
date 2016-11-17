#![feature(iter_max_by)]
#![feature(proc_macro)]

extern crate rand;
extern crate rayon;
extern crate nalgebra as na;
extern crate image;
extern crate clap;
extern crate pbr;

#[macro_use] extern crate serde_derive;
extern crate serde;
extern crate bincode;

mod multilayer_perceptron;
mod activation_func;
mod img;
mod validators;
mod args;

use std::collections::HashMap;
use multilayer_perceptron::MultilayerPerceptron;
use std::fs::File;

fn main() {
    let matches = args::get();
    if let Some(matches) = matches.subcommand_matches("learn") {
        learn(matches);
    }
    if let Some(matches) = matches.subcommand_matches("check") {
        check(matches);
    }
}

fn get_img_and_label<P: AsRef<std::path::Path>>(p: P) -> (Vec<f64>, String) {
    let image = img::get_pixels(&p);
    let label = p.as_ref().file_name().unwrap().to_str().unwrap().split("_").next().unwrap();
    (image, String::from(label))
}

fn check(matches: &clap::ArgMatches<'static>) {
    let check_dir = matches.value_of("check-dataset").unwrap();
    let in_net = matches.value_of("in-net").unwrap();


    let NetFile(perc, neuron_to_label) = bincode::serde::deserialize_from(
        &mut File::open(in_net).expect("couldn't open input net"), bincode::SizeLimit::Infinite
    ).expect("couldn't decode net file");

    print!("Loading checking dataset from {}... ", check_dir);
    let check_imgs: Vec<_> = {
        let paths = std::fs::read_dir(check_dir).unwrap();
        paths.map(|p| get_img_and_label(p.unwrap().path())).collect()
    };
    println!("Loaded!");

    let mut correct = 0;
    use na::Iterable;
    for &(ref img, ref label) in &check_imgs {
        let (out, _) = perc.feed_forward(img);
        let decoded = &neuron_to_label[&out.iter().enumerate()
            .max_by(|a, b|
                a.1.partial_cmp(b.1).unwrap()
            ).unwrap().0];
        if decoded == label {
            correct += 1;
        }
    }

    println!("{} / {} correct", correct, check_imgs.len());
}

fn learn(matches: &clap::ArgMatches<'static>) {
    let learn_dir = matches.value_of("learn-dataset").unwrap();
    let sample: f64 = matches.value_of("learn-sample").unwrap().parse().unwrap();
    let max_epochs: u64 = matches.value_of("max-epochs").unwrap().parse().unwrap();
    let learning_rate: f64 = matches.value_of("learning-rate").unwrap().parse().unwrap();
    let parallel = matches.value_of("no-parallel").is_none();
    let input_net = matches.value_of("in-net");
    let out_net = matches.value_of("out-net");

    use std::fs;
    let paths = fs::read_dir(learn_dir).unwrap();

    use std::collections::HashSet;

    print!("Loading learning dataset from {}... ", learn_dir);
    let imgs: Vec<_> = paths.map(|p| get_img_and_label(p.unwrap().path())).collect();
    let learning_labels: HashSet<&str> = imgs.iter()
        .map(|&(_, ref label)| label.as_str()).collect();
    println!("Loaded!");

    let sample_amt = (sample * imgs.len() as f64) as usize;

    use multilayer_perceptron::MultilayerPerceptron;
    use activation_func::Tanh;

    let (mut perc, labels) = if let Some(path) = input_net {
        use bincode::SizeLimit::*;
        let NetFile(perc, labels) = bincode::serde::deserialize_from(
            &mut File::open(path).expect("couldn't open input net"), Infinite
        ).expect("couldn't decode net file");
        (perc, Some(labels))
    } else {
        (MultilayerPerceptron::new(
            learning_rate,
            imgs[0].0.len(),
            &[
                (100, Tanh(1.0).into()), // TODO: make that configurable
                (learning_labels.len(), Tanh(1.0).into())
            ]
        ), None)
    };

    let neuron_to_label;
    {
        let label_to_neuron = if let Some(l) = labels {
            neuron_to_label = l;
            let ltn: HashMap<&str, (usize, Vec<f64>)> = neuron_to_label.iter().map(|(&i, s)| {
                let target = (0..learning_labels.len())
                    .map(|j| if j == i { 1.0 } else { 0.0 }).collect();
                (s.as_str(), (i, target))
            }).collect();
            ltn
        } else {
            let ltn: HashMap<&str, (usize, Vec<f64>)> = learning_labels.iter().enumerate()
                .map(|(i, &label)| {
                    let target = (0..learning_labels.len())
                        .map(|j| if j == i { 1.0 } else { 0.0 }).collect();
                    (label, (i, target))
                }).collect();

            neuron_to_label = ltn.iter()
                .map(|(&label, &(i, _))| {
                    (i, label.into())
                }).collect();
            ltn
        };

        println!("Learning...");
        use pbr::ProgressBar;
        let mut pb = ProgressBar::new(max_epochs);
        if parallel {
            for _ in 0..max_epochs {
                let sample: Vec<(&[f64], &[f64])> = rand::sample(
                    &mut rand::thread_rng(),
                    imgs.iter().map(|&(ref image, ref label)| (&image[..], &label_to_neuron[label.as_str()].1[..])),
                    sample_amt
                );
                perc.learn_batch(&sample);
                pb.inc();
            }
        } else {
            for _ in 0..max_epochs {
                let sample: Vec<(&[f64], &[f64])> = rand::sample(
                    &mut rand::thread_rng(),
                    imgs.iter().map(|&(ref image, ref label)| (&image[..], &label_to_neuron[label.as_str()].1[..])),
                    sample_amt
                );
                perc.learn_batch_no_parallel(&sample);
                pb.inc();
            }
        }
        pb.finish_println("Finished learning!\n");
    }

    if let Some(path) = out_net {
        use bincode::SizeLimit::*;
        bincode::serde::serialize_into(
            &mut File::create(&path).expect("couldn't create the output net file"),
            &NetFile(perc, neuron_to_label),
            Infinite
        ).expect("couldn't save the net");
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct NetFile(MultilayerPerceptron, HashMap<usize, String>);

#[test]
fn tests_for_raport() {
    let dir = "res/Sieci Neuronowe";

    for hidden in 2..121 {
        print!("{} ", hidden);
        raport(dir, dir, 0.2, 50, 0.5, 10, hidden);
    }

}

fn raport(learn_dir: &str, check_dir: &str, sample: f64, max_epochs: usize, learning_rate: f64, skip: usize, hidden: usize) {
    let check_imgs_vec;

    //    let learn_dir = "res/Sieci Neuronowe";
    //    let check_dir = learn_dir;
    //    let sample = 0.2;
    //    let max_epochs = 10000;
    //    let learning_rate = 0.5;

    use std::fs;
    let paths = fs::read_dir(learn_dir).unwrap();

    use std::collections::HashSet;

    //    print!("Loading learning dataset from {}... ", learn_dir);
    let imgs: Vec<_> = paths.map(|p| get_img_and_label(p.unwrap().path())).collect();
    let learning_labels: HashSet<&str> = imgs.iter()
        .map(|&(_, ref label)| label.as_str()).collect();
    //    println!("Loaded!");

    //    print!("Loading checking dataset from {}... ", check_dir);
    let check_imgs = if check_dir == learn_dir {
        &imgs
    } else {
        let paths = fs::read_dir(check_dir).unwrap();
        check_imgs_vec = paths.map(|p| get_img_and_label(p.unwrap().path())).collect();
        &check_imgs_vec
    };
    let checking_labels: HashSet<&str> = check_imgs.iter()
        .map(|&(_, ref label)| label.as_str()).collect();
    //    println!("Loaded!");

    if learning_labels != checking_labels {
        panic!("Labels from learning and checking datasets are different");
    }

    let sample_amt = (sample * imgs.len() as f64) as usize;

    use multilayer_perceptron::MultilayerPerceptron;
    use activation_func::Tanh;
    let mut perc = MultilayerPerceptron::new(
        learning_rate,
        imgs[0].0.len(),
        &[
            (hidden, Tanh(1.0).into()), // TODO: make that configurable
            (learning_labels.len(), Tanh(1.0).into())
        ]
    );

    //    println!("Learning...");

    use std::collections::HashMap;
    let label_to_neuron: HashMap<&str, (usize, Vec<f64>)> = learning_labels.iter().enumerate().map(|(i, &label)| {
        let target = (0..learning_labels.len()).map(|j| if j == i { 1.0 } else { 0.0 }).collect();
        (label, (i, target))
    }).collect();

    let neuron_to_label: HashMap<usize, &str> = label_to_neuron.iter().map(|(&label, &(i, _))| {
        (i, label)
    }).collect();

    //    use pbr::ProgressBar;
    //    let mut pb = ProgressBar::new(max_epochs);

    for i in 0..max_epochs {
        let sample: Vec<(&[f64], &[f64])> = rand::sample(
            &mut rand::thread_rng(),
            imgs.iter().map(|&(ref image, ref label)| (&image[..], &label_to_neuron[label.as_str()].1[..])),
            sample_amt
        );
        perc.learn_batch(&sample);
        if i % skip == 0 {
//            print!("{}\t", i);
//            println_correct(&check_imgs, &perc, &neuron_to_label);
        }

        //        pb.inc();
    }

    //    pb.finish_println("Finished learning!\n");

        println_correct(&check_imgs, &perc, &neuron_to_label);
}

fn println_correct(check_imgs: &[(Vec<f64>, String)], perc: &MultilayerPerceptron, neuron_to_label: &HashMap<usize, &str>) {
    let mut correct = 0;
    use na::Iterable;
    for &(ref img, ref label) in check_imgs {
        let (out, _) = perc.feed_forward(img);
        let decoded = neuron_to_label[&out.iter().enumerate()
            .max_by(|a, b|
                a.1.partial_cmp(b.1).unwrap()
            ).unwrap().0];
        if decoded == label {
            correct += 1;
        }
    }

    println!("{}", correct);
}

#[test] fn and_test() {
    use activation_func::Tanh;
    let mut perc = MultilayerPerceptron::new(
        0.5,
        2,
        &[
            (2, Tanh(1.0).into()), // TODO: make that configurable
            (1, Tanh(1.0).into())
        ]
    );

    let examples = [
        (&[1.0, 1.0][..], &[1.0][..]),
        (&[1.0, 0.0][..], &[0.0][..]),
        (&[0.0, 1.0][..], &[1.0][..]),
        (&[0.0, 0.0][..], &[0.0][..]),
    ];

    for _ in 0..1000 {
        perc.learn_batch(&examples);
    }

    let mut err = 0.0;
    use na::Iterable;
    for &(ref example, ref target) in &examples {
        let (out, _) = perc.feed_forward(example);
        err += (out[0] - target[0]).abs();
    }

    println!("{}", err);
}