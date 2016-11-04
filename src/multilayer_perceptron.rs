use rand;
use activation_func::{ActivationFunction, Tanh, ActivationFunctionEnum};
use na::{DMatrix, DVector};

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

    fn activate(&self, inputs: DVector<f64>) -> DVector<f64> {
        let mut res = inputs;
        res *= &self.neurons;
        for i in 0..res.len() {
            res[i] = self.activation_function.function(res[i]);
        }
        res
    }
}

pub struct MultilayerPerceptron {
    hidden_layers: Vec<Layer>,
    output_layer: Layer,
    learning_rate: f64
}

impl MultilayerPerceptron {
    fn new(
        learning_rate: f64,
        inputs: usize,
        hidden_layers: &[(usize, ActivationFunctionEnum)],
        outputs: (usize, ActivationFunctionEnum)
    ) -> MultilayerPerceptron {
        let mut hidden = Vec::with_capacity(hidden_layers.len());
        let mut prev_layer_size = inputs + 1;

        for i in 0..hidden_layers.len() {
            hidden.push(Layer::new(
                hidden_layers[i].1,
                DMatrix::from_fn(prev_layer_size, hidden_layers[i].0, |_, _| rand::random()))
            );
            prev_layer_size = hidden_layers[i].0;
        }

        let output = Layer::new(
            outputs.1,
            DMatrix::from_fn(prev_layer_size, outputs.0, |_, _| rand::random())
        );

        MultilayerPerceptron {
            hidden_layers: hidden,
            output_layer: output,
            learning_rate: learning_rate
        }
    }

    fn feed_forward(&self, input: &[f64]) -> (DVector<f64>, Vec<DVector<f64>>) {
        let layer_outputs = Vec::with_capacity(self.hidden_layers.len() + 2);
        let mut signal = make_dvector_with_bias(input);
        layer_outputs.push(signal);

        for layer in &self.hidden_layers {
            signal = layer.activate(signal);
            layer_outputs.push(signal);
        }

        let out = self.output_layer.activate(signal);
        layer_outputs.push(out);
        (out, layer_outputs)
    }

    fn backpropagate(&mut self, input: &[f64], target: &[f64]) {
        let expected_output = DVector::from_slice(target.len(), target);
        let (final_out, steps) = self.feed_forward(input);
        if final_out.len() != expected_output.len() {
            panic!("expected_output has wrong length: expected: {}, given: {}",
                   final_out.len(), expected_output.len())
        }

        let error = expected_output - final_out;

        let input_delta_weights = self.learning_rate * () * final_out;
    }
}

fn hadamard_prod(x: DVector<f64>, y: &DVector<f64>) -> DVector<f64> {
    for i in 0..x.len() {
        x.at[i] *= y.at[i]
    }
    x
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
            (2, Tanh(1.0).into())
        ],
        (2, Tanh(1.0).into()) // output layer (number of neurons, activation function)
    );
    let out = perc.feed_forward(&inputs);
    assert!(out.0.len() == 2);
}
