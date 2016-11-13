#![feature(iter_max_by)]

extern crate rand;
extern crate rayon;
extern crate nalgebra as na;
extern crate image;
extern crate clap;
extern crate pbr;

mod multilayer_perceptron;
mod activation_func;
mod img;
mod validators;
mod args;

fn main() {
    let check_imgs_vec; // temp var to hold images from the checking dataset. Declared here because
    // compiler complained if it was declared after `learning_labels`

    let matches = args::get();

    let learn_dir = matches.value_of("learn-dataset").unwrap();
    let check_dir = matches.value_of("check-dataset").unwrap_or(learn_dir);
    let sample: f64 = matches.value_of("learn-sample").unwrap().parse().unwrap();
    let max_epochs: u64 = matches.value_of("max-epochs").unwrap().parse().unwrap();
    let learning_rate: f64 = matches.value_of("learning-rate").unwrap().parse().unwrap();
    let parallel = matches.value_of("no-parallel").is_none();

    use std::fs;
    let paths = fs::read_dir(learn_dir).unwrap();

    use std::collections::HashSet;

    print!("Loading learning dataset from {}... ", learn_dir);
    let imgs: Vec<_> = paths.map(|p| get_img_and_label(p.unwrap().path())).collect();
    let learning_labels: HashSet<&str> = imgs.iter()
        .map(|&(_, ref label)| label.as_str()).collect();
    println!("Loaded!");

    print!("Loading checking dataset from {}... ", check_dir);
    let check_imgs = if check_dir == learn_dir {
        &imgs
    } else {
        let paths = fs::read_dir(check_dir).unwrap();
        check_imgs_vec = paths.map(|p| get_img_and_label(p.unwrap().path())).collect();
        &check_imgs_vec
    };
    let checking_labels: HashSet<&str> = check_imgs.iter()
        .map(|&(_, ref label)| label.as_str()).collect();
    println!("Loaded!");

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
            (100, Tanh(1.0).into()), // TODO: make that configurable
            (learning_labels.len(), Tanh(1.0).into())
        ]
    );

    println!("Learning...");

    use std::collections::HashMap;
    let label_to_neuron: HashMap<&str, (usize, Vec<f64>)> = learning_labels.iter().enumerate().map(|(i, &label)| {
        let target = (0..learning_labels.len()).map(|j| if j == i { 1.0 } else { 0.0 }).collect();
        (label, (i, target))
    }).collect();

    let neuron_to_label: HashMap<usize, &str> = label_to_neuron.iter().map(|(&label, &(i, _))| {
        (i, label)
    }).collect();

    use pbr::ProgressBar;
    let mut pb = ProgressBar::new(max_epochs);
    if parallel {
        for i in 0..max_epochs {
            let sample: Vec<(&[f64], &[f64])> = rand::sample(
                &mut rand::thread_rng(),
                imgs.iter().map(|&(ref image, ref label)| (&image[..], &label_to_neuron[label.as_str()].1[..])),
                sample_amt
            );
            perc.learn_batch(&sample);
            pb.inc();
        }
    } else {
        for i in 0..max_epochs {
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

    println!("{} / {} correct", correct, check_imgs.len());
}

fn get_img_and_label<P: AsRef<std::path::Path>>(p: P) -> (Vec<f64>, String) {
    let image = img::get_pixels(&p);
    let label = p.as_ref().file_name().unwrap().to_str().unwrap().split("_").next().unwrap();
    (image, String::from(label))
}