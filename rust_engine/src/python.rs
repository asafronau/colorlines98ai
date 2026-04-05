/// PyO3 bindings: drop-in replacement for game.board.ColorLinesGame.

use pyo3::prelude::*;
use numpy::{PyArray2, PyReadonlyArray2, PyArrayMethods};
use crate::board::*;
use crate::game::ColorLinesGame;
use crate::rng::SimpleRng;

#[pyclass(name = "RustColorLinesGame")]
pub struct PyColorLinesGame {
    inner: ColorLinesGame,
}

#[pymethods]
impl PyColorLinesGame {
    #[new]
    #[pyo3(signature = (seed=0))]
    fn new(seed: u64) -> Self {
        PyColorLinesGame {
            inner: ColorLinesGame::new(seed),
        }
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Clone game state. If rng_seed is provided, the clone gets a fresh RNG.
    #[pyo3(signature = (rng_seed=None))]
    fn clone_game(&self, rng_seed: Option<u64>) -> Self {
        let rng = match rng_seed {
            Some(s) => SimpleRng::new(s),
            None => SimpleRng::new(0),
        };
        PyColorLinesGame {
            inner: self.inner.clone_with_rng(rng),
        }
    }

    /// Board as 9x9 numpy array (int8). Returns a copy.
    fn get_board<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<i8>> {
        let mut flat = [0i8; 81];
        for r in 0..BOARD_SIZE {
            for c in 0..BOARD_SIZE {
                flat[r * BOARD_SIZE + c] = self.inner.board[r][c];
            }
        }
        numpy::PyArray1::from_slice(py, &flat)
            .reshape([9, 9])
            .unwrap()
    }

    /// Set board from 9x9 numpy array.
    fn set_board(&mut self, board: PyReadonlyArray2<i8>) {
        let arr = board.as_array();
        for r in 0..BOARD_SIZE {
            for c in 0..BOARD_SIZE {
                self.inner.board[r][c] = arr[[r, c]];
            }
        }
    }

    #[getter]
    fn score(&self) -> i32 {
        self.inner.score
    }

    #[getter]
    fn turns(&self) -> i32 {
        self.inner.turns
    }

    #[getter]
    fn game_over(&self) -> bool {
        self.inner.game_over
    }

    #[getter]
    fn next_balls(&self) -> Vec<((usize, usize), i8)> {
        self.inner.next_balls_tuples()
    }

    /// Execute a move with full validation.
    /// Returns dict: {valid, score, cleared, game_over}
    fn move_ball(&mut self, sr: usize, sc: usize, tr: usize, tc: usize) -> PyResult<PyObject> {
        let (valid, score, cleared, game_over) = self.inner.move_ball(sr, sc, tr, tc);
        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("valid", valid)?;
            dict.set_item("score", score)?;
            dict.set_item("cleared", cleared)?;
            dict.set_item("game_over", game_over)?;
            Ok(dict.into())
        })
    }

    /// Execute a move known to be legal (skip validation).
    fn trusted_move(&mut self, sr: usize, sc: usize, tr: usize, tc: usize) {
        self.inner.trusted_move(sr, sc, tr, tc);
    }

    /// Source mask: 9x9 int8 array, 1 where ball can move.
    fn get_source_mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<i8>> {
        let mask = get_source_mask(&self.inner.board);
        let mut flat = [0i8; 81];
        for r in 0..BOARD_SIZE {
            for c in 0..BOARD_SIZE {
                flat[r * BOARD_SIZE + c] = mask[r][c];
            }
        }
        numpy::PyArray1::from_slice(py, &flat)
            .reshape([9, 9])
            .unwrap()
    }

    /// Target mask: 9x9 int8 array, 1 where source can reach.
    fn get_target_mask<'py>(&self, py: Python<'py>, sr: usize, sc: usize) -> Bound<'py, PyArray2<i8>> {
        let labels = label_empty_components(&self.inner.board);
        let mask = get_target_mask(&labels, sr, sc);
        let mut flat = [0i8; 81];
        for r in 0..BOARD_SIZE {
            for c in 0..BOARD_SIZE {
                flat[r * BOARD_SIZE + c] = mask[r][c];
            }
        }
        numpy::PyArray1::from_slice(py, &flat)
            .reshape([9, 9])
            .unwrap()
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyColorLinesGame>()?;
    m.add_function(wrap_pyfunction!(py_calculate_score, m)?)?;
    Ok(())
}

#[pyfunction]
fn py_calculate_score(num_balls: usize) -> i32 {
    calculate_score(num_balls)
}
