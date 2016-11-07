pub trait ActivationFunction: Send + Sync + Copy {
    fn function(&self, x: f64) -> f64;
    fn derivative(&self, x: f64) -> f64;
}

#[derive(Debug, Copy, Clone)]
pub struct Sigmoid(pub f64);

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

    fn derivative(&self, x: f64) -> f64 {
        self.function(x) * (1.0 - self.function(x))
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Linear(pub f64);

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

    fn derivative(&self, _x: f64) -> f64 {
        self.0
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Tanh(pub f64);

impl Tanh {
    fn new(a: f64) -> Tanh { Tanh(a) }
}

impl Default for Tanh {
    fn default() -> Self {
        Tanh::new(1.0)
    }
}

impl ActivationFunction for Tanh {
    fn function(&self, x: f64) -> f64 {
        (x * self.0).tanh()
    }

    fn derivative(&self, x: f64) -> f64 {
        fn sech(x: f64) -> f64 { 1.0 / x.cosh() }
        sech(x) * sech(x)
    }
}

#[derive(Debug, Copy, Clone)]
pub enum ActivationFunctionEnum {
    Sigmoid(Sigmoid),
    Linear(Linear),
    Tanh(Tanh)
}

impl ActivationFunction for ActivationFunctionEnum {
    fn function(&self, x: f64) -> f64 {
        use self::ActivationFunctionEnum::*;
        match self {
            &Sigmoid(f) => f.function(x),
            &Linear(f) => f.function(x),
            &Tanh(f) => f.function(x)
        }
    }

    fn derivative(&self, x: f64) -> f64 {
        use self::ActivationFunctionEnum::*;
        match self {
            &Sigmoid(f) => f.derivative(x),
            &Linear(f) => f.derivative(x),
            &Tanh(f) => f.derivative(x)
        }
    }
}

impl From<Sigmoid> for ActivationFunctionEnum {
    fn from(s: Sigmoid) -> Self {
        ActivationFunctionEnum::Sigmoid(s)
    }
}

impl From<Linear> for ActivationFunctionEnum {
    fn from(l: Linear) -> Self {
        ActivationFunctionEnum::Linear(l)
    }
}

impl From<Tanh> for ActivationFunctionEnum {
    fn from(t: Tanh) -> Self {
        ActivationFunctionEnum::Tanh(t)
    }
}