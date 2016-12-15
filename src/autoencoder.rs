use multilayer_perceptron::{MultilayerPerceptron, SparsityParams};
use activation_func::*;
use rand;
use img::get_img_and_label;
use std::fs;
use img;
use clap;
use mnist;
use rayon::prelude::*;

pub fn run(matches: &clap::ArgMatches<'static>) -> Result<(), &'static str> {
    let epoch_count: u64 = matches.value_of("epoch-count").and_then(|x| x.parse().ok()).unwrap_or(5000);
    let is_mnist = matches.is_present("mnist");
    let sample_ratio: f64 = matches.value_of("sample").and_then(|x| x.parse().ok()).unwrap_or(0.01);

    let (w, h, images_own) = if is_mnist {
        let images_own: Vec<_> = mnist::MnistDigits::default_training_set().unwrap();
        (28, 28, images_own)
    } else {
        let paths: Vec<_> = fs::read_dir("res/Sieci Neuronowe").unwrap().map(|p| p.unwrap().path()).collect();
        let images_own: Vec<_> = paths.iter().map(|p| get_img_and_label(p)).collect();
        (7, 10, images_own)
    };

    let images: Vec<_> = images_own.iter().map(|&(ref x, _)| (&x[..], &x[..])).collect();
    let mut autoencoder = MultilayerPerceptron::new(0.3, images[0].0.len(), &[
        (matches.value_of("hidden-neurons").and_then(|x| x.parse().ok()).unwrap_or(25), Sigmoid(1.0).into()),
        (images[0].0.len(), Sigmoid(10.0).into())
    ]);

    autoencoder.sparsity_params = Some(SparsityParams {
        sparsity: matches.value_of("sparsity").and_then(|x| x.parse().ok()).unwrap_or(0.05),
        penalty_factor: matches.value_of("penalty-factor").and_then(|x| x.parse().ok()).unwrap_or(0.8),
    });

    let sample_size = (images.len() as f64 * sample_ratio) as usize;

    use pbr::ProgressBar;
    let mut pbr = ProgressBar::new(epoch_count);

    for i in 0..epoch_count {
        let sample: Vec<(&[f64], &[f64])> = rand::sample(&mut rand::thread_rng(), images.iter().cloned(), sample_size);
        autoencoder.learn_batch(&sample);
        pbr.inc();
    }

    let outs: Vec<_> = images.iter().map(|i| autoencoder.feed_forward(&i.0[..]).0.at).collect();
    for (ii, i) in outs.into_iter().enumerate() {
        if is_mnist && ii % 1000 != 0 {
            continue;
        }
        use std::path::Path;
        let name = format!("out{}.png", ii);
        let p = Path::new("autoencoded").join(&name);
        img::save(&i, w, h, p);
    }

    extract_features(&autoencoder, w, h);

    let error = calc_err(&autoencoder, &images);
    pbr.finish_println(&format!("error: {0:>5.2}  ", error));

    Ok(())
}

fn extract_features(net: &MultilayerPerceptron, w: u32, h: u32) {
    use na::{Column, Iterable, Row};
    let ref hidden_layer = net.layers[1];

    for i in 0..hidden_layer.weights.nrows() {
        let row = hidden_layer.weights.row(i);
        let sum: f64 = row.iter().map(|&x| x * x).sum();
        let l: f64 = sum.sqrt();
        let xs: Vec<_> = (0..hidden_layer.weights.ncols()).map(|j|
            1.0 - (hidden_layer.weights[(i, j)] / l)
        ).collect();
        let name = format!("autoencoded/feature{}.png", i);
        img::save(&xs, w, h, &name);
    }
}

fn calc_err(network: &MultilayerPerceptron, data: &[(&[f64], &[f64])]) -> f64 {
    use na::{DVector, norm};
    use std::iter::FromIterator;
    let sum: f64 = data.par_iter().map(|&(ref img, ref label)| {
        let out = network.feed_forward(img).0;
        let label = DVector::from_iter(label.iter().cloned());
        norm(&(label - out))
    }).sum();
    sum / (data.len() as f64)
}

macro_rules! trace {
    ($e:expr) => { {println!(concat!(stringify!($e), " = {:?}"), $e)} };
}

#[test]
fn autoencoder_raport() {
    //// LOAD IMAGES ////
    let paths: Vec<_> = fs::read_dir("res/Sieci Neuronowe").unwrap().map(|p| p.unwrap().path()).collect();
    let images_own: Vec<_> = paths.iter().map(|p| get_img_and_label(p)).collect();
    let images: Vec<_> = images_own.iter().map(|&(ref x, _)| (&x[..], &x[..])).collect();

    let autoencoder = MultilayerPerceptron::new(0.3, images[0].0.len(), &[
        (50, Sigmoid(1.0).into()),
        (images[0].0.len(), Sigmoid(1.0).into())
    ]);

    for (i, &(sparsity, penalty_factor, feature_idx, img_idx)) in [
        (0.05, 0.0, 20, 508),
        (0.05, 0.1, 20, 508),
        (0.05, 0.3, 20, 508),
        (0.05, 0.5, 20, 508),
        (0.05, 0.7, 20, 508),
        (0.05, 0.9, 20, 508),
        (0.05, 1.1, 20, 508),
        (0.05, 1.3, 20, 508),
        (0.05, 1.5, 20, 508),
        (0.05, 1.7, 20, 508),
        (0.05, 1.9, 20, 508),
    ].into_iter().enumerate() {
        println!("next:");
        trace!((i, sparsity, penalty_factor));

        use std::fs::{self, DirBuilder};

        let path = format!("raport/{}", i);
        DirBuilder::new()
            .recursive(true)
            .create(&path).unwrap();

        let mut autoencoder: MultilayerPerceptron = autoencoder.clone();

        autoencoder.sparsity_params = Some(SparsityParams {
            sparsity: sparsity,
            penalty_factor: penalty_factor,
        });

        let sample_size = (images.len() as f64 * 0.1) as usize;

        let epoch_count: u64 = 5_000;

        for j in 0..epoch_count {
            let sample: Vec<(&[f64], &[f64])> = rand::sample(&mut rand::thread_rng(), images.iter().cloned(), sample_size);
            autoencoder.learn_batch(&sample);

            if j % 100 == 0 {
                let error = calc_err(&autoencoder, &images);
                println!("{:.2}", error);

                dump_feature(&autoencoder, feature_idx, 7, 10, &format!("{}/feat_epoch_{:05}.png", &path, j));
                let image = autoencoder.feed_forward(&images[img_idx].0[..]).0.at;
                img::save(&image, 7, 10, &format!("{}/img_epoch_{:05}.png", &path, j))
            }
        }
    }
}

fn dump_feature(net: &MultilayerPerceptron, i: usize, w: u32, h: u32, name: &str) {
    use na::{Column, Iterable, Row};
    let ref hidden_layer = net.layers[1];
    let row = hidden_layer.weights.row(i);
    let sum: f64 = row.iter().map(|&x| x * x).sum();
    let l: f64 = sum.sqrt();
    let xs: Vec<_> = (0..hidden_layer.weights.ncols()).map(|j|
        1.0 - (hidden_layer.weights[(i, j)] / l)
    ).collect();
    img::save(&xs, w, h, &name);
}