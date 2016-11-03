pub trait ActivationFunction {
    fn function(&self, x: f64) -> f64;
    fn dereviative(&self, x: f64) -> f64;
}

pub struct Sigmoid(f64);

impl Sigmoid {
    pub fn new(a: f64) -> Self { Sigmoid(a) }
}

impl Default for Sigmoid {
    fn default() -> Self {
        Sigmoid::new(1.0)
    }
}

impl ActivationFunction for Sigmoid {
    fn function(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x * self.0).exp())
    }

    fn dereviative(&self, x: f64) -> f64 {
        self.function(x) * (1.0 - self.function(x))
    }
}

pub struct Linear(f64);

impl Linear {
    pub fn new(a: f64) -> Self { Linear(a) }
}

impl Default for Linear {
    fn default() -> Self {
        Linear::new(1.0)
    }
}

impl ActivationFunction for Linear {
    fn function(&self, x: f64) -> f64 {
        self.0 * x
    }

    fn dereviative(&self, _x: f64) -> f64 {
        self.0
    }
}