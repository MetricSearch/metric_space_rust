use anyhow::Result;
use bits::{bsp, bsp_distance, bsp_similarity, f32_data_to_bsp, f32_data_to_cubeoct_bitrep, f32_embedding_to_bsp, whamming_distance};
use bitvec_simd::BitVecSimd;
use metrics::euc;
use ndarray::{Array1, ArrayView1, Axis};
use rayon::prelude::*;
use std::collections::HashSet;
use std::rc::Rc;
use std::time::Instant;
use wide::u64x4;
use dao::{Dao};
use dao::csv_dao_loader::{dao_from_csv_dir};
use utils::arg_sort_2d;
//use divan::Bencher;

fn main() -> Result<()> {
    tracing::info!("Loading mf dino data...");
    let num_queries = 10_000;
    let num_data = 1_000_000 - num_queries;

    let dao_f32: Rc<Dao<Array1<f32>>> = Rc::new(dao_from_csv_dir(
        "/Volumes/Data/RUST_META/mf_dino2_csv/",
        num_data,
        num_queries,
    )?);
    // just take 1 query

    let queries = dao_f32.get_queries();
    let data = dao_f32.get_data();

    for i in (0..10) {
        println!( "Data [0][{}]: {}", i, data[0][i] );
    }

    let bsp_0 = f32_embedding_to_bsp::<2>(&data[0],200);

    println!("Data_0 has {} bits | XORed = {} bits", bsp_0.ones.count_ones() + bsp_0.negative_ones.count_ones(), bsp_0.ones.xor_cloned(&bsp_0.negative_ones).count_ones());

    // println!("Data_1 has {} bits | XORed = {} bits", data_1.ones.count_ones() + data_1.negative_ones.count_ones(), data_1.ones.xor_cloned(&data_1.negative_ones).count_ones());
    //
    println!("Object 0 bitrep ones: {:?} | negative ones: {:?}", bsp_0.ones, bsp_0.negative_ones);

    // println!( "Smoking 0-0 distance {} similarity: {} ", bsp_distance::<2>(&data_0, &data_0), bsp_similarity::<2>(&data_0, &data_0) ); // two girls  --> 1024 + 200
    // println!( "data 1-1 leaves distance {} similarity: {} ", bsp_distance::<2>(data_1, &data_1), bsp_similarity::<2>(data_1, &data_1) ); // smoking girl and leaves.
    // println!( "data 2-2 leaves distance {} similarity: {} ", bsp_distance::<2>(data_2, &data_2), bsp_similarity::<2>(data_2, &data_2) ); // smoking girl and leaves.
    // println!( "Smoking0-585 distance {} similarity: {} ", bsp_distance::<2>(data_585585, &data_0), bsp_similarity::<2>(data_585585, &data_0) ); // two girls
    // println!( "Smoking 0-1 leaves distance {} similarity: {} ", bsp_distance::<2>(data_1, &data_0), bsp_similarity::<2>(data_1, &data_0) ); // smoking girl and leaves.
    // println!( "Query 0-1 data distance {} similarity: {}  ", bsp_distance::<2>(query_0, &data_1), bsp_similarity::<2>(query_0, &data_1) ); // badness from other

    Ok(())
}
