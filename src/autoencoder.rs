use mnist;
use multilayer_perceptron::MultilayerPerceptron;
use activation_func::*;
use rand;
use ::get_img_and_label;
use std::fs;
use img;

pub fn run() -> Result<(), &'static str> {
    let epoch_count = 5000;

    let paths: Vec<_> = fs::read_dir("res/Sieci Neuronowe").unwrap().map(|p| p.unwrap().path()).collect();
    let images_own: Vec<_> = paths.iter().map(|p| get_img_and_label(p)).collect();
    let images: Vec<_> = images_own.iter().map(|&(ref x, _)| (&x[..], &x[..])).collect();
    let mut autoencoder = MultilayerPerceptron::new(0.3, 7 * 10, &[
        (20, Tanh(1.0).into()),
        (7 * 10, Sigmoid(50.0).into())
    ]);

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

    Ok(())
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