use generic_array::{ArrayLength, GenericArray};
use typenum::consts::{U8, U2};
use hlist::LayerHList;

#[derive(Default, Copy, Clone, Debug)]
pub struct StaticNeuron<N: ArrayLength<f64>> {
    weights: GenericArray<f64, N>
}

pub struct DynamicNeuron {
    weights: Vec<f64>
}

pub trait Neuron {
    fn get_weights(&self) -> &[f64];
    fn get_mut_weights(&mut self) -> &mut [f64];
}

impl<N: ArrayLength<f64>> Neuron for StaticNeuron<N> {
    fn get_weights(&self) -> &[f64] {
        &self.weights
    }

    fn get_mut_weights(&mut self) -> &mut [f64] {
        &mut self.weights
    }
}

impl Neuron for DynamicNeuron {
    fn get_weights(&self) -> &[f64] {
        &self.weights
    }

    fn get_mut_weights(&mut self) -> &mut [f64] {
        &mut self.weights
    }
}

#[derive(Clone, Debug)]
pub struct StaticLayer<N: ArrayLength<StaticNeuron<Prev>>, Prev: ArrayLength<f64>> {
    neurons: GenericArray<StaticNeuron<Prev>, N>
}

impl<N: ArrayLength<StaticNeuron<Prev>>, Prev: ArrayLength<f64>> Default for StaticLayer<N, Prev> {
    fn default() -> Self {
        StaticLayer { neurons: GenericArray::<StaticNeuron<Prev>, N>::new() }
    }
}

#[test]
fn test_layer_length() {
    let x: StaticLayer<U8, U2> = Default::default();
    assert!(x.neurons.len() == 8)
}

#[derive(Default, Clone, Debug)]
pub struct DynamicLayer {
    neurons: Vec<DynamicNeuron>
}

pub trait Layer {
    type NeuronT: Neuron;

    fn get_neurons(&self) -> &[Self::NeuronT];
    fn get_mut_neurons(&mut self) -> &mut [Self::NeuronT];

    fn size(&self) -> usize {
        self.get_neurons().len()
    }
}

impl Layer for DynamicLayer {
    type NeuronT = DynamicNeuron;
    fn get_neurons(&self) -> &[Self::NeuronT] {
        &self.neurons
    }

    fn get_mut_neurons(&mut self) -> &mut [Self::NeuronT] {
        &mut self.neurons
    }
}

impl<N, Prev> Layer for StaticLayer<N, Prev>
where N: ArrayLength<StaticNeuron<Prev>>, Prev: ArrayLength<f64> {
    type NeuronT = StaticNeuron<Prev>;
    fn get_neurons(&self) -> &[Self::NeuronT] {
        &self.neurons
    }

    fn get_mut_neurons(&mut self) -> &mut [Self::NeuronT] {
        &mut self.neurons
    }
}

pub struct MultilayerPerceptron<InputLayer, HiddenLayers, OutputLayer> where
    InputLayer: Layer,
    HiddenLayers: LayerHList,
    OutputLayer: Layer
{
    input_layer: InputLayer,
    hidden_layers: HiddenLayers,
    output_layer: OutputLayer
}


