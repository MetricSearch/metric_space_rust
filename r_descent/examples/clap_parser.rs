use anyhow::Result;
use bits::{f32_data_to_cubic_bitrep, whamming_distance};
use bitvec_simd::BitVecSimd;
use dao::csv_dao_matrix_loader::dao_matrix_from_csv_dir;
use dao::{Dao, DaoMatrix};
use metrics::euc;
use ndarray::Array1;
use r_descent_matrix::{get_nn_table2_m, initialise_table_m};
use std::collections::HashSet;
use std::fs::File;
use std::io::BufReader;
use std::rc::Rc;
use std::time::Instant;
use clap::Parser;
use utils::dot_product_f32;

/// clap parser
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Name of the person to greet
    #[arg(short, long)]
    name: String,

    /// Number of times to greet
    #[arg(short, long, default_value_t = 1)]
    count: u8,
}

fn main() -> Result<()> {
    let args = Args::parse();
    Ok(())
}