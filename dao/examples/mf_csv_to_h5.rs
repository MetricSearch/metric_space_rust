use bits::{
    bsp_distance, bsp_similarity, f32_data_to_bsp, f32_data_to_cubeoct_bitrep, whamming_distance,
    EvpBits,
};
use bitvec_simd::BitVecSimd;
use anyhow::Result;
use dao::csv_dao_loader::dao_from_csv_dir;
use dao::hdf5_dao_loader::hdf5_f32_write;
use dao::Dao;
use metrics::euc;
use ndarray::{Array1, ArrayView1, Axis};
use rayon::prelude::*;
use std::collections::HashSet;
use std::rc::Rc;
use std::time::Instant;
use utils::arg_sort_2d;
use wide::u64x4;
//use divan::Bencher;

fn main() -> Result<()> {
    tracing::info!("Loading mf dino data...");
    let num_queries = 10_000;
    let num_data = 1_000_000 - num_queries;

    let dest_data_path = "/Volumes/Data/RUST_META/mf_dino2.h5";

    let dao: Rc<Dao<Array1<f32>>> = Rc::new(dao_from_csv_dir(
        "/Volumes/Data/RUST_META/mf_dino2_csv/",
        num_data,
        num_queries,
    )?);

    hdf5_f32_write(
        dest_data_path,
        &dao.meta.name,
        &dao.meta.description,
        &dao.meta.normed.to_string(),
        &dao.embeddings,
    )?;

    Ok(())
}
