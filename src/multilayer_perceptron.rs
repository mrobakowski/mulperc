use rand;
use activation_func::{ActivationFunction, Sigmoid};
use na::{DMatrix, DVector, Dot};
use rayon::prelude::*;
use std;

fn make_dvector_with_bias(x: &[f64]) -> DVector<f64> {
    let mut i = DVector::from_slice(x.len(), x);
    i.at.push(1.0);
    i
}

#[derive(Clone, Debug)]
pub struct Layer {
    neurons: DMatrix<f64>
}

impl Layer {
    fn activate<A>(&self, f: A, inputs: DVector<f64>) -> DVector<f64> where A: ActivationFunction {
        inputs * &self.neurons
    }
}

pub struct MultilayerPerceptron {
    hidden_layers: Vec<Layer>,
    output_layer: Layer
}

impl MultilayerPerceptron {
    fn new(inputs: usize, hidden_layers: &[usize], outputs: usize) -> MultilayerPerceptron {
        let mut hidden = Vec::with_capacity(hidden_layers.len());
        let mut prev_layer_size = inputs + 1;

        for i in 0..hidden_layers.len() {
            hidden.push(Layer {
                neurons: DMatrix::from_fn(prev_layer_size, hidden_layers[i], |_, _| rand::random())
            });
            prev_layer_size = hidden_layers[i];
        }

        let output = Layer {
            neurons: DMatrix::from_fn(prev_layer_size, outputs, |_, _| rand::random())
        };

        MultilayerPerceptron {
            hidden_layers: hidden,
            output_layer: output,
        }
    }

    fn feed_forward<A>(&self, f: A, inputs: &[f64]) -> DVector<f64> where A: ActivationFunction {
        let mut signal = make_dvector_with_bias(inputs);

        for layer in &self.hidden_layers {
            signal = layer.activate(f, signal);
        }

        self.output_layer.activate(f, signal)
    }
}

#[test]
fn test_feedforward_matrices_sizes() {
    let inputs = [1.0, 2.0, 3.0, -1.0];
    let perc = MultilayerPerceptron::new(inputs.len(), &[2, 3, 6, 2], 2);
    let out = perc.feed_forward(Sigmoid(1.0), &inputs);
    assert!(out.len() == 2);
}
