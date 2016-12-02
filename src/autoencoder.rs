use mnist;
use multilayer_perceptron::MultilayerPerceptron;
use activation_func::Tanh;
use rand;
use ::get_img_and_label;
use std::fs;

pub fn run() -> Result<(), &'static str> {
    let epoch_count = 10000;

    let mnist_own: Vec<_> = {
        let paths = fs::read_dir("res/Sieci Neuronowe").unwrap();
        paths.map(|p| get_img_and_label(p.unwrap().path())).collect()
    };//mnist::MnistDigits::default_training_set()?;
    let mnist: Vec<_> = mnist_own.iter().map(|&(ref x, _)| (&x[..], &x[..])).collect();
    //    let mnist_test = mnist::MnistDigits::default_test_set()?;
    let mut autoencoder = MultilayerPerceptron::new(0.3, 7 * 10, &[
        (20, Tanh(1.0).into()),
        (7 * 10, Tanh(1.0).into())
    ]);

    let sample_size = (mnist.len() as f64 * 0.2) as usize;

    use pbr::ProgressBar;
    let mut pbr = ProgressBar::new(epoch_count);

    for _ in 0..epoch_count {
        let sample: Vec<(&[f64], &[f64])> = rand::sample(&mut rand::thread_rng(), mnist.iter().cloned(), sample_size);
        autoencoder.learn_batch(&sample);
        let error = calc_err(&autoencoder, &mnist);
        pbr.message(&format!("error: {0:>5.2}  ", error));
        pbr.inc();
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