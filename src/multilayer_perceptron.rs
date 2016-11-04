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
        let mut res = inputs * &self.neurons;
        for i in 0..res.len() {
            res[i] = self.activation_function.function(res[i]);
        }
        res
    }
}

pub struct MultilayerPerceptron {
    hidden_layers: Vec<Layer>,
    output_layer: Layer
}

impl MultilayerPerceptron {
    fn new(
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
        }
    }

    fn feed_forward(&self, inputs: &[f64]) -> DVector<f64> {
        let mut signal = make_dvector_with_bias(inputs);

        for layer in &self.hidden_layers {
            signal = layer.activate(signal);
        }

        self.output_layer.activate(signal)
    }
}

#[test]
fn test_feedforward_matrices_sizes() {
    let inputs = [1.0, 2.0, 3.0, -1.0];
    let perc = MultilayerPerceptron::new(
        inputs.len(), // number of inputs
        &[ // hidden layers (number of neurons, activation function)
            (2, Tanh(1.0).into()),
            (3, Tanh(1.0).into()),
            (6, Tanh(1.0).into()),
            (2, Tanh(1.0).into())
        ],
        (2, Tanh(1.0).into())
    );
    let out = perc.feed_forward(&inputs);
    assert!(out.len() == 2);
}
