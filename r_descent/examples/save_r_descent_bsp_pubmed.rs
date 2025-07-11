use anyhow::Result;
use bits::container::Simd256x2;
use bits::EvpBits;
use chrono::Utc;
use dao::csv_dao_loader::dao_from_csv_dir;
use dao::hdf5_to_dao_loader::hdf5_f32_to_bsp_load;
use dao::pubmed_hdf5_gt_loader::hdf5_pubmed_gt_load;
use dao::Dao;
use ndarray::{s, Array1, Array2, ArrayView1};
use r_descent::{get_nn_table2_bsp, initialise_table_bsp, IntoRDescent, RDescent};
use std::collections::HashSet;
use std::fs::File;
use std::io::BufWriter;
use std::rc::Rc;
use std::time::Instant;
use utils::distance_f32;

fn main() -> Result<()> {
    let start = Instant::now();
    let utc = Utc::now();
    println!("Started at {:?}", utc);
    println!("Loading Pubmed data...");

    let f_name = "/Volumes/Data/sisap_challenge_25/pubmed/benchmark-dev-pubmed23.h5";

    let num_queries = 10_000;
    const ALL_RECORDS: usize = 0;
    const NUM_VERTICES: usize = 200;

    let dao_bsp = Rc::new(
        hdf5_f32_to_bsp_load::<Simd256x2, 384>(f_name, ALL_RECORDS, num_queries, NUM_VERTICES)
            .unwrap(),
    );

    let queries = dao_bsp.get_queries();
    let data = dao_bsp.get_data();

    println!(
        "Pubmed data size: {} queries size: {}",
        data.len(),
        queries.len()
    );

    let start_post_load = Instant::now();

    let num_neighbours = 10;
    let chunk_size = 200;
    let rho = 1.0;
    let delta = 0.01;
    let reverse_list_size = 32;

    let descent =
        dao_bsp
            .clone()
            .into_rdescent(num_neighbours, reverse_list_size, chunk_size, delta);

    let end = Instant::now();

    println!(
        "Finished (including load time in {} s",
        (end - start).as_secs()
    );
    println!(
        "Finished (post load time) in {} s",
        (end - start_post_load).as_secs()
    );

    println!("Saving NN table to _scratch/pubmed_table_10.bin ...");

    let f = BufWriter::new(File::create("_scratch/pubmed_table_10.bin").unwrap());
    let _ = bincode::serialize_into(f, &descent);

    Ok(())
}
