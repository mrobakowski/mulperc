use rand;
use activation_func::{ActivationFunction, Tanh, ActivationFunctionEnum};
use na::{DMatrix, DVector, Transpose, Iterable, Outer, Shape};

fn make_dvector_with_bias(x: &[f64]) -> DVector<f64> {
    let mut i = DVector::from_slice(x.len(), x);
    i.at.push(1.0);
    i
}

#[derive(Clone, Debug)]
pub struct Layer {
    neurons: DMatrix<f64>,
    activation_function: ActivationFunctionEnum
}

impl Layer {
    fn new<A>(activation_function: A, weights: DMatrix<f64>) -> Layer where A: Into<ActivationFunctionEnum> {
        Layer {
            neurons: weights,
            activation_function: activation_function.into()
        }
    }

    fn activate(&self, inputs: &DVector<f64>) -> DVector<f64> {
        let mut res = inputs * &self.neurons;
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
        let (final_out, mut steps) = self.feed_forward(input);
        let num_steps = steps.len();

        if final_out.len() != expected_output.len() {
            panic!("expected_output has wrong length: expected: {}, given: {}",
                   final_out.len(), expected_output.len())
        }

        let last_layer = &self.layers[self.layers.len() - 1];

        let f_prim_z: DVector<_> = final_out.iter()
            .map(|&x| last_layer.activation_function.dereviative(x)).collect();

        let error = final_out - expected_output;
        let output_layer_delta = error * f_prim_z;

        println!("after error calc");

        let mut deltas = Vec::with_capacity(self.layers.len() + 1);
        deltas.push(output_layer_delta);

        for i in 0..self.layers.len() {
            let layer_index = self.layers.len() - 1 - i; // we start at the last layer
            let current_layer: &Layer = &self.layers[layer_index];
            let outputs_from_current_hidden_layer = &steps[num_steps - 1 - i];

            let f_prim_z: DVector<_> = outputs_from_current_hidden_layer.iter()
                .map(|&x| current_layer.activation_function.dereviative(x)).collect();
            println!("after f_prim_z");

            let neurons_transposed = current_layer.neurons.transpose();
            println!("neurons transposed: {:?}", neurons_transposed.shape());
            let delta = {
                let prev_delta = &deltas[i];
                println!("prev delta: {:?}", prev_delta.shape());
                let d1 = prev_delta * neurons_transposed;
                println!("d1 size: {:?}, f_prim_z size: {:?}", d1.shape(), f_prim_z.shape());
                d1 * f_prim_z
            };
            println!("after delta");
            deltas.push(delta);
        }

        deltas.into_iter().zip(steps.iter().rev())
            .map(|(d, s)| d.outer(s)).collect()
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
            (2, Tanh(1.0).into()),
            (3, Tanh(1.0).into()),
            (6, Tanh(1.0).into()),
            (5, Tanh(1.0).into()),
            (2, Tanh(1.0).into())
        ]
    );
    let deltas = perc.backpropagate(&inputs, &[1.0, 0.0]);
    println!("{:?}", deltas);
}
