use rand;
use activation_func::{ActivationFunction, Tanh, ActivationFunctionEnum};
use na::{DMatrix, DVector, Transpose, Outer, IterableMut};
use std::ops::Deref;
use rayon::prelude::*;

fn make_dvector_with_bias(x: &[f64]) -> DVector<f64> {
    let mut i = DVector::from_slice(x.len(), x);
    i.at.push(1.0);
    i
}

#[derive(Clone, Debug)]
pub struct Layer {
    weights: DMatrix<f64>,
    activation_function: ActivationFunctionEnum
}

impl Layer {
    fn new<A>(activation_function: A, weights: DMatrix<f64>) -> Layer where A: Into<ActivationFunctionEnum> {
        Layer {
            weights: weights,
            activation_function: activation_function.into()
        }
    }

    fn net(&self, inputs: &DVector<f64>) -> DVector<f64> {
        inputs * &self.weights
    }

    fn activate(&self, inputs: &DVector<f64>) -> DVector<f64> {
        let mut res = self.net(inputs);
        for i in 0..res.len() {
            res[i] = self.activation_function.function(res[i]);
        }
        res
    }
}

pub struct MultilayerPerceptron {
    layers: Vec<Layer>,
    learning_rate: f64
}

impl MultilayerPerceptron {
    fn new(
        learning_rate: f64,
        inputs: usize,
        layers: &[(usize, ActivationFunctionEnum)]
    ) -> MultilayerPerceptron {
        let mut l = Vec::with_capacity(layers.len());
        let mut prev_layer_size = inputs + 1;

        for i in 0..layers.len() {
            l.push(Layer::new(
                layers[i].1,
                DMatrix::from_fn(prev_layer_size, layers[i].0, |_, _| rand::random()))
            );
            prev_layer_size = layers[i].0;
        }

        MultilayerPerceptron {
            layers: l,
            learning_rate: learning_rate
        }
    }

    fn feed_forward(&self, input: &[f64]) -> (DVector<f64>, Vec<DVector<f64>>) {
        let mut layer_inputs = Vec::with_capacity(self.layers.len() + 1);
        let mut signal = make_dvector_with_bias(input);

        for layer in &self.layers {
            let new_signal = layer.activate(&signal);
            layer_inputs.push(signal);
            signal = new_signal;
        }

            (signal, layer_inputs)
    }

    /// Returns delta weights
    fn backpropagate(&self, input: &[f64], target: &[f64]) -> Vec<DMatrix<f64>> {
        let expected_output = DVector::from_slice(target.len(), target);
        let (final_out, steps) = self.feed_forward(input);
        let num_steps = steps.len();

        if final_out.len() != expected_output.len() {
            panic!("expected_output has wrong length: expected: {}, given: {}",
                   final_out.len(), expected_output.len())
        }

        let last_layer: &Layer = self.layers.last().unwrap();

        let mut f_prim_z: DVector<_> = last_layer.net(steps.last().unwrap());
        for x in f_prim_z.iter_mut() {
            *x = last_layer.activation_function.derivative(*x);
        }

        let error = final_out - expected_output;
        let output_layer_delta = error * f_prim_z;

        let mut deltas = Vec::with_capacity(self.layers.len());
        deltas.push(output_layer_delta);

        for i in 0..self.layers.len() - 1 {
            let layer_index = self.layers.len() - 1 - i;
            let inputs_to_current_hidden_layer = &steps[num_steps - 2 - i];
            let l: &Layer = &self.layers[layer_index - 1];

            let mut f_prim_z: DVector<_> = l.net(inputs_to_current_hidden_layer);
            for x in f_prim_z.iter_mut() {
                *x = l.activation_function.derivative(*x);
            }

            let neurons_transposed = self.layers[layer_index].weights.transpose();
            let delta = &deltas[i] * neurons_transposed * f_prim_z;
            deltas.push(delta);
        }

        assert!(deltas.len() == steps.len());

        deltas.into_iter().rev().zip(steps.iter())
            .map(|(d, s)| d.outer(s)).collect()
    }

    fn learn_batch<I, T>(&mut self, batch: &[(I, T)])
        where I: Deref<Target = [f64]> + Sync, T: Deref<Target = [f64]> + Sync {
        let mut batch_delta = batch.par_iter()
            .map(|&(ref i, ref t)| Some(self.backpropagate(i.deref(), t.deref())))
            .reduce(|| None, |acc, v_opt| {
                acc.and_then(|mut old_v: Vec<DMatrix<f64>>| {
                    v_opt.as_ref().map(|v| {
                        for (i, x) in old_v.iter_mut().enumerate() { *x += &v[i] }
                        old_v
                    })
                }).or(v_opt)
            }).unwrap();

        for x in &mut batch_delta {
            for el in x.as_mut_vector() {
                *el /= batch.len() as f64
            }
        }

        for (l, d) in self.layers.iter_mut().zip(batch_delta.iter()) {
            l.weights += d
        }
    }
}

#[test]
fn test_feedforward_matrices_sizes() {
    let inputs = [1.0, 2.0, 3.0, -1.0];
    let perc = MultilayerPerceptron::new(
        0.01, // learning rate
        inputs.len(), // number of inputs
        &[// hidden layers (number of neurons, activation function)
            (2, Tanh(1.0).into()),
            (3, Tanh(1.0).into()),
            (6, Tanh(1.0).into()),
            (5, Tanh(1.0).into()),
            (2, Tanh(1.0).into()) // output layer (number of neurons, activation function)
        ]
    );
    let out = perc.feed_forward(&inputs);
    assert!(out.0.len() == 2);
}

#[test]
fn test_backpropagation_matrices_sizes() {
    let inputs = [1.0, 2.0, 3.0, -1.0];
    let perc = MultilayerPerceptron::new(
        0.01,
        inputs.len(),
        &[
            (4, Tanh(1.0).into()),
            (3, Tanh(1.0).into()),
            (2, Tanh(1.0).into())
        ]
    );
    let deltas = perc.backpropagate(&inputs, &[0.0, 1.0]);
    println!("{:?}", deltas.len());
}
