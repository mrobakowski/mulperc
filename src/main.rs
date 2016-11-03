extern crate generic_array;
extern crate typenum;

mod hlist;
mod multilayer_perceptron;
mod activation_func;

fn main() {
    println!("Hello, world!");

    use std::sync::{Mutex, Arc};
    let mut x = Arc::new(Mutex::new(0));

    use std::thread;
    let y1 = x.clone();
    let t1 = thread::spawn(move || {
        let mut guard = y1.lock().unwrap();
        *guard += 1;

        println!("1: {}", *guard);
    });

    let y2 = x.clone();
    let t2 = thread::spawn(move || {
        let mut guard = y2.lock().unwrap();
        *guard += 1;

        println!("2: {}", *guard);
    });

    {
        let xx = x.lock().unwrap();
        println!("3: {}", *xx);
    }

    t1.join();
    t2.join();
}