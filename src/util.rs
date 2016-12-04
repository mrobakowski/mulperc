use std::error::Error;
use std::fmt::Display;
use std::fmt;
use std::convert::From;

#[derive(Debug, Copy, Clone)]
pub struct DontCare;

impl<E: Error> From<E> for DontCare {
    fn from(_: E) -> Self {
        DontCare
    }
}

#[macro_export]
macro_rules! catch {
    ($($tts:tt)*) => { (|| $($tts)*)() };
}

#[macro_export]
macro_rules! ignore_err {
    ($($tts:tt)*) => { {let x: ::std::result::Result<_, $crate::util::DontCare> = (|| {let x = {$($tts)*}?; ::std::result::Result::Ok(x)})(); x.ok()} };
}