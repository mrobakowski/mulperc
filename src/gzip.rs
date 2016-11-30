extern crate flate2;
extern crate libc;

use self::flate2::read::GzDecoder;
use std::fs::File;
use std::io::Read;
use std::io;
use self::libc::*;
use std::mem;

/// Struct to decompress gzip streams.
pub struct GzipData {
    v: Vec<u8>,
    idx: usize
}

impl GzipData {
    pub fn from_file(fname: &str) -> Result<GzipData, &'static str> {
        let mut r: Vec<u8> = Vec::new();
        GzDecoder::new(File::open(fname).map_err(|_| "Could not open file")?)
            .map_err(|_| "Invalid gzip header.")?
            .read_to_end(&mut r)
            .map_err(|_| "Could not unzip data.")?;

        Ok(GzipData {
            v: r,
            idx: 0
        })
    }

    // TODO test
    pub fn from_buf(v: Vec<u8>) -> GzipData {

        GzipData {
            v: v,
            idx: 0
        }
    }

    /// Returns the uncompressed data.
    pub fn into_bytes(self) -> Vec<u8> { self.v }

    /// Returns the length of the uncomressed data.
    pub fn len(&self) -> usize { self.v.len() }

    /// Returns an iterator over the uncompressed data. TODO test
    pub fn iter<'a>(&'a self) -> impl Iterator<Item = &'a u8> { self.v.iter().skip(self.idx) }

    // TODO test
    pub fn buf<'b>(&'b self) -> &'b [u8] { &self.v.split_at(self.idx).1 }
}

/// Implementation of the `Read` trait for GzipData.
impl Read for GzipData {

    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {

        if self.idx >= self.v.len() {
            return Ok(0);
        }

        let n = buf.len();
        let c = copy_memory(
            buf,
            self.v.split_at(self.idx).1,
            n
        );
        self.idx += c;
        Ok(c)
    }
}

pub fn copy_memory<T: Copy>(dst: &mut [T], src: &[T], n: usize) -> usize {
    use std::cmp::min;
    let c = min(min(dst.len(), src.len()), n);
    unsafe {
        memcpy(
            dst.as_ptr()              as *mut c_void,
            src.as_ptr()              as *const c_void,
            (c * mem::size_of::<T>()) as size_t
        );
    }
    c
}