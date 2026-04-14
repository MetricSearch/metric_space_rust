/* A copy of challenge1 to test the Hamiltonians */

use anyhow::Result;
use clap::Parser;
use ndarray::{Array1, ArrayView, Ix1};

use half::f16;
use hamiltonians::{get_cycle_lengths, get_cycle_lookup_table, get_vertex_number, make_pascal};
use std::time::Instant;
use utils::non_nan::NonNan;

use std::ops::{Add, Div, Mul, Sub};

#[derive(Clone, Copy, Debug)]
struct BigExpFloat {
    mantissa: f64,
    exponent: i64,
}

impl BigExpFloat {
    fn new(mantissa: f64, exponent: i64) -> Self {
        Self { mantissa, exponent }.normalize()
    }

    fn normalize(mut self) -> Self {
        if self.mantissa == 0.0 {
            self.exponent = 0;
            return self;
        }

        while self.mantissa.abs() >= 2.0 {
            self.mantissa /= 2.0;
            self.exponent += 1;
        }
        while self.mantissa.abs() < 1.0 {
            self.mantissa *= 2.0;
            self.exponent -= 1;
        }
        self
    }

    /// Align two numbers to the same exponent (for + and -)
    fn align(self, other: Self) -> (Self, Self) {
        if self.exponent > other.exponent {
            let shift = (self.exponent - other.exponent) as i32;
            let scaled = other.mantissa / 2f64.powi(shift);
            (
                self,
                BigExpFloat {
                    mantissa: scaled,
                    exponent: self.exponent,
                },
            )
        } else {
            let shift = (other.exponent - self.exponent) as i32;
            let scaled = self.mantissa / 2f64.powi(shift);
            (
                BigExpFloat {
                    mantissa: scaled,
                    exponent: other.exponent,
                },
                other,
            )
        }
    }
}

impl BigExpFloat {
    fn to_string_sci(&self) -> String {
        if self.mantissa == 0.0 {
            return "0".to_string();
        }

        // log10(x) = log10(mantissa) + exponent * log10(2)
        let log10 =
            self.mantissa.abs().log10() + (self.exponent as f64) * std::f64::consts::LOG10_2;

        let exp10 = log10.floor();
        let mant10 = 10f64.powf(log10 - exp10);

        let sign = if self.mantissa < 0.0 { "-" } else { "" };

        format!("{}{}e{}", sign, mant10, exp10 as i64)
    }
}

impl BigExpFloat {
    fn from_f64(x: f64) -> Self {
        if x == 0.0 {
            return Self {
                mantissa: 0.0,
                exponent: 0,
            };
        }

        // Extract exponent via log2
        let exp = x.abs().log2().floor() as i64;
        let mant = x / 2f64.powi(exp as i32);

        Self {
            mantissa: mant,
            exponent: exp,
        }
        .normalize()
    }
}

// Addition
impl Add for BigExpFloat {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let (a, b) = self.align(other);
        BigExpFloat {
            mantissa: a.mantissa + b.mantissa,
            exponent: a.exponent,
        }
        .normalize()
    }
}

// Subtraction
impl Sub for BigExpFloat {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let (a, b) = self.align(other);
        BigExpFloat {
            mantissa: a.mantissa - b.mantissa,
            exponent: a.exponent,
        }
        .normalize()
    }
}

// Multiplication
impl Mul for BigExpFloat {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        BigExpFloat {
            mantissa: self.mantissa * other.mantissa,
            exponent: self.exponent + other.exponent,
        }
        .normalize()
    }
}

// Division
impl Div for BigExpFloat {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        BigExpFloat {
            mantissa: self.mantissa / other.mantissa,
            exponent: self.exponent - other.exponent,
        }
        .normalize()
    }
}

/// Pascal's triangle lookup table.
/// Returns a 2D array of size (x rows) x (d cols),
/// analogous to MATLAB's makePascal_rel(x, d).
pub fn make_pascalBig(rows: usize, cols: usize) -> Vec<Vec<BigExpFloat>> {
    let mut tri = vec![vec![BigExpFloat::from_f64(1.0); cols]; rows];
    // First row and first column are all 1s - set above.
    for i in 1..rows {
        for j in 1..cols {
            tri[i][j] = tri[i - 1][j] + tri[i][j - 1];
        }
    }
    tri
}

fn main() -> Result<()> {
    pretty_env_logger::formatted_timed_builder()
        .filter_level(log::LevelFilter::Trace)
        .init();

    const D: usize = 1024;
    const NON_ZEROS: usize = 512;

    // Build Pascal triangle
    let pas_tri: Vec<Vec<BigExpFloat>> = make_pascalBig(D + 2, D + 2);

    log::debug!("last row of pas_tri: {:?}", pas_tri[pas_tri.len() - 1]);

    // // Build cycle lengths and lookup tables
    // let c_lengths: Vec<usize> = get_cycle_lengths_fast_rel(NON_ZEROS);
    // let mut tables: Vec<Vec<Vec<bool>>> = Vec::with_capacity(NON_ZEROS);
    // for xi in 1..=NON_ZEROS {
    //     tables.push(get_cycle_lookup_table(c_lengths[xi - 1], xi, &pas_tri));
    // }

    Ok(())
}
