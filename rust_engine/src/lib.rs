pub mod rng;
pub mod board;
pub mod game;

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn colorlines98(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register(m)
}
