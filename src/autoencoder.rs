use multilayer_perceptron::{MultilayerPerceptron, SparsityParams};
use activation_func::*;
use rand;
use img::get_img_and_label;
use std::fs;
use img;

pub fn run() -> Result<(), &'static str> {
    let epoch_count = 5000;

    let paths: Vec<_> = fs::read_dir("res/Sieci Neuronowe").unwrap().map(|p| p.unwrap().path()).collect();
    let images_own: Vec<_> = paths.iter().map(|p| get_img_and_label(p)).collect();
    let images: Vec<_> = images_own.iter().map(|&(ref x, _)| (&x[..], &x[..])).collect();
    let mut autoencoder = MultilayerPerceptron::new(0.3, 7 * 10, &[
        (50, Sigmoid(1.0).into()),
        (7 * 10, Sigmoid(50.0).into())
    ]);

    autoencoder.sparsity_params = Some(SparsityParams {
        sparsity: 0.05,
        penalty_factor: 0.8,
    });

    let sample_size = (images.len() as f64 * 0.1) as usize;

    use pbr::ProgressBar;
    let mut pbr = ProgressBar::new(epoch_count);

    for _ in 0..epoch_count {
        let sample: Vec<(&[f64], &[f64])> = rand::sample(&mut rand::thread_rng(), images.iter().cloned(), sample_size);
        autoencoder.learn_batch(&sample);
        let error = calc_err(&autoencoder, &images);
        pbr.message(&format!("error: {0:>5.2}  ", error));
        pbr.inc();
    }

    let outs: Vec<_> = images.iter().map(|i| autoencoder.feed_forward(&i.0[..]).0.at).collect();
    for (i, p) in outs.into_iter().zip(paths.iter()) {
        use std::path::Path;
        let p = Path::new("autoencoded").join(p);
        img::save(&i, 7, 10, p);
    }

    extract_features(&autoencoder);

    Ok(())
}

fn extract_features(net: &MultilayerPerceptron) {
    use na::{Column, Iterable, Row};
    let ref hidden_layer = net.layers[1];
    for i in 0..hidden_layer.weights.ncols() {
        let col = hidden_layer.weights.column(i);
        let sum: f64 = col.iter().map(|&x| x * x).sum();
        let l: f64 = sum.sqrt();
        let xs: Vec<_> = (0..hidden_layer.weights.nrows()).map(|j|
            hidden_layer.weights[(j, i)] / l
        ).collect();
        let name = format!("autoencoded/feature{}.png", i);
        img::save(&xs, 7, 10, &name);
    }
}

fn calc_err(network: &MultilayerPerceptron, data: &[(&[f64], &[f64])]) -> f64 {
    use na::{DVector, norm};
    use std::iter::FromIterator;
    let sum: f64 = data.iter().map(|&(ref img, ref label)| {
        let out = network.feed_forward(img).0;
        let label = DVector::from_iter(label.iter().cloned());
        norm(&(label - out))
    }).sum();
    sum / (data.len() as f64)
}

#[test]
fn test_autoencode() {
    run().unwrap();
}