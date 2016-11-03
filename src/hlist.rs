use multilayer_perceptron::{Layer, StaticLayer, DynamicLayer};
use typenum::consts::{U8, U2, U1};

pub trait LayerHList {
    fn get_layer(&self) -> Option<&Layer>;
    fn get_next_or_nil(&self) -> &LayerHList;
    fn iter<'a>(&'a self) -> Iter<'a> where Self: Sized {
        (self as &LayerHList).into_iter()
    }
}

impl<'a> IntoIterator for &'a LayerHList {
    type Item = &'a Layer;
    type IntoIter = Iter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        Iter { hlist: self }
    }
}

pub trait NotEmpty {}

#[repr(C)]
pub struct Cons<T: Layer, Tail: LayerHList> {
    head: T,
    tail: Tail
}

impl<T, Tail> Default for Cons<T, Tail> where T: Layer + Default, Tail: LayerHList + Default {
    fn default() -> Self {
        Cons {
            head: Default::default(),
            tail: Default::default()
        }
    }
}

#[derive(Debug, Default, Copy, Clone)]
pub struct Nil;

macro_rules! LayerList {
    ($t: ty, $($ts: ty),*) => (Cons<$t, LayerList![$($ts),*]>);
    ($t: ty) => (Cons<$t, Nil>);
    () => (Nil);
}

#[test]
fn macro_and_iter_test() {
    let x: LayerList![
        StaticLayer<U8>,
        StaticLayer<U2>,
        DynamicLayer,
        StaticLayer<U1>
    ] = Default::default();

    let mut iter = x.iter();
    assert!(iter.next().unwrap().get_neurons().len() == 8);
    assert!(iter.next().unwrap().get_neurons().len() == 2);
    assert!(iter.next().unwrap().get_neurons().len() == 0);
    assert!(iter.next().unwrap().get_neurons().len() == 1);
    assert!(iter.next().is_none());
}

impl LayerHList for Nil {
    fn get_layer(&self) -> Option<&Layer> {
        None
    }
    fn get_next_or_nil(&self) -> &LayerHList {
        self
    }
}

impl<T: Layer, Tail: LayerHList> LayerHList for Cons<T, Tail> {
    fn get_layer(&self) -> Option<&Layer> {
        Some(&self.head)
    }
    fn get_next_or_nil(&self) -> &LayerHList {
        &self.tail
    }
}

impl<T: Layer, Tail: LayerHList> NotEmpty for Cons<T, Tail> {}

pub struct Iter<'a> {
    hlist: &'a LayerHList
}

impl<'a> Iterator for Iter<'a> {
    type Item = &'a Layer;
    fn next(&mut self) -> Option<Self::Item> {
        let layer = self.hlist.get_layer();
        self.hlist = self.hlist.get_next_or_nil();
        layer
    }
}