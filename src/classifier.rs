use rand;
use std::fs::File;
use bincode;
use clap;
use multilayer_perceptron::NetFile;
use std::collections::HashMap;
use mnist;
use std;
use img::get_img_and_label;

pub fn check(matches: &clap::ArgMatches<'static>) {
    let check_dir = matches.value_of("check-dataset").unwrap();
    let in_net = matches.value_of("in-net").unwrap();

    let NetFile(perc, neuron_to_label) = bincode::serde::deserialize_from(
        &mut File::open(in_net).expect("couldn't open input net"), bincode::SizeLimit::Infinite
    ).expect("couldn't decode net file");

    print!("Loading checking dataset from {}... ", check_dir);
    let check_imgs: Vec<_> = if check_dir == "mnist" {
        mnist::MnistDigits::default_test_set().unwrap()
    } else {
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

pub fn learn(matches: &clap::ArgMatches<'static>) {
    let learn_dir = matches.value_of("learn-dataset").unwrap();
    let sample: f64 = matches.value_of("learn-sample").unwrap().parse().unwrap();
    let max_epochs: u64 = matches.value_of("max-epochs").unwrap().parse().unwrap();
    let learning_rate: f64 = matches.value_of("learning-rate").unwrap().parse().unwrap();
    let parallel = !matches.is_present("no-parallel");
    println!("parallel: {}", parallel);
    let input_net = matches.value_of("in-net");
    let out_net = matches.value_of("out-net");

    use std::fs;
    use std::collections::HashSet;

    print!("Loading learning dataset from {}... ", learn_dir);
    let imgs: Vec<_> = if learn_dir == "mnist" {
        mnist::MnistDigits::default_training_set().unwrap()
    } else {
        let paths = fs::read_dir(learn_dir).unwrap();
        paths.map(|p| get_img_and_label(p.unwrap().path())).collect()
    };

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
                (200, Tanh(1.0).into()), // TODO: make that configurable
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
