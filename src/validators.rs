pub fn path_exists(s: String) -> Result<(), String> {
    use std::path::*;
    let meta = Path::new(&s).metadata().map_err(|e| e.to_string())?;
    if meta.is_dir() {
        Ok(())
    } else {
        Err(format!("{} is not a directory", s))
    }
}

pub fn file_exists(s: String) -> Result<(), String> {
    use std::path::*;
    let meta = Path::new(&s).metadata().map_err(|e| e.to_string())?;
    if meta.is_file() {
        Ok(())
    } else {
        Err(format!("{} is not a file", s))
    }
}

pub fn str_is_float(s: String) -> Result<(), String> {
    use std::str::FromStr;
    f64::from_str(&s).map(|_| ()).map_err(|_| format!("{} is not a float", s))
}

pub fn str_is_integer(s: String) -> Result<(), String> {
    use std::str::FromStr;
    i64::from_str(&s).map(|_| ()).map_err(|_| format!("{} is not an integer", s))
}