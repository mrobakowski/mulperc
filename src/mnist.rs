use gzip::GzipData;
use std::io::Read;

pub struct MnistDigits;

impl MnistDigits {

    fn from_high_endian(arr: &[u8]) -> u64 {
        arr.iter().fold(0, |acc, val| acc * 256 + (*val as u64))
    }

    fn read_u32<T: Read>(src: &mut T) -> Result<u32, &'static str> {
        let mut buf: [u8; 4] = [0; 4];

        match src.read(&mut buf) {
            Ok(4) => Ok(MnistDigits::from_high_endian(&buf) as u32),
            _     => Err("Could not read data."),
        }
    }

    fn read_labels(fname: &str) -> Result<Vec<u8>, &'static str> {

        let mut data = GzipData::from_file(fname)?;

        if MnistDigits::read_u32(&mut data)? != 8 * 256 + 1 {
            return Err("Invalid magic number.");
        }

        let n = MnistDigits::read_u32(&mut data)?;

        let l: Vec<u8> = data.iter().cloned().collect();
        if l.len() != n as usize {
            return Err("Invalid number of items.");
        }

        if !l.iter().all(|&x| x <= 9) {
            return Err("Found invalid values for labels.");
        }

        Ok(l)
    }

    fn read_examples(fname: &str) -> Result<Vec<Vec<f64>>, &'static str> {

        let mut data = GzipData::from_file(fname)?;

        if MnistDigits::read_u32(&mut data)? != 8 * 256 + 3 {
            return Err("Invalid magic number.");
        }

        let n = MnistDigits::read_u32(&mut data)?;

        let rows = MnistDigits::read_u32(&mut data)?;
        let cols = MnistDigits::read_u32(&mut data)?;
        if rows != 28 || cols != 28 {
            return Err("Invalid number of rows or columns.");
        }

        let v = data.buf();
        if v.len() != (n * 28 * 28) as usize {
            return Err("Could not read data.");
        }

        let res: Vec<Vec<f64>> = v.chunks(28 * 28)
            .map(|chunk| chunk.iter().map(|&byte| (byte as f64 / u8::max_value() as f64)).collect())
            .collect();

        Ok(res)
    }

    pub fn from(vectors_fname: &str, labels_fname: &str) -> Result<Vec<(Vec<f64>, String)>, &'static str> {
        let labels = MnistDigits::read_labels(labels_fname)?;
        let values = MnistDigits::read_examples(vectors_fname)?;

        Ok(values.into_iter().zip(labels.into_iter().map(|label| label.to_string())).collect())
    }

    fn path(fname: &str) -> Result<String, &'static str> {
        use std::path::*;
        let mut pbf = PathBuf::new();
        pbf.push("res/mnist");
        pbf.push(fname);
        pbf.as_path().to_str().map(|x| x.to_string()).ok_or("Could not create path to mnist dataset")
    }

    pub fn default_training_set() -> Result<Vec<(Vec<f64>, String)>, &'static str> {
        let features = MnistDigits::path("train-images-idx3-ubyte.gz")?;
        let labels = MnistDigits::path("train-labels-idx1-ubyte.gz")?;
        MnistDigits::from(&features, &labels)
    }

    pub fn default_test_set() -> Result<Vec<(Vec<f64>, String)>, &'static str> {
        let features = MnistDigits::path("t10k-images-idx3-ubyte.gz")?;
        let labels = MnistDigits::path("t10k-labels-idx1-ubyte.gz")?;
        MnistDigits::from(&features, &labels)
    }
}