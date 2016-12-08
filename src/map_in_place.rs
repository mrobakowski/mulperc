use std;

struct Sizes<FROM, TO> {
    _from: std::marker::PhantomData<FROM>,
    _to: std::marker::PhantomData<TO>,
}

trait Compatible {}

impl<T> Compatible for Sizes<T, T> {}

impl Compatible for Sizes<u64, f64> {}

impl Compatible for Sizes<i64, f64> {}

impl Compatible for Sizes<u64, i64> {}

trait MapInPlace<FROM> {
    fn mip<F, TO>(self, f: F) -> Vec<TO> where F: Fn(FROM) -> TO, Sizes<FROM, TO>: Compatible;
}

impl<FROM> MapInPlace<FROM> for Vec<FROM> {
    fn mip<F, TO>(mut self, f: F) -> Vec<TO> where F: Fn(FROM) -> TO, Sizes<FROM, TO>: Compatible {
        let vv: &mut [TO] = unsafe { std::mem::transmute((&mut self) as &mut [FROM]) };
        for (i, x) in self.iter_mut().enumerate() {
            let xx = std::mem::replace(x, unsafe { std::mem::uninitialized() });
            vv[i] = f(xx);
        }
        unsafe { std::mem::transmute(self) }
    }
}

#[test]
fn test_mip() {
    let x = vec![1u64, 2, 3];
    let y = x.mip(|x| x as f64);
    assert!(y == vec![1.0f64, 2.0, 3.0]);
}