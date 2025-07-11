use anyhow::Result;
use bits::{hamming_distance, hamming_distance_as_f32};
use bitvec_simd::BitVecSimd;
use dao::convert_f32_to_cubic::to_cubic_dao;
use dao::csv_dao_loader::dao_from_csv_dir;
use dao::Dao;
use descent::Descent;
use ndarray::Array1;
use std::rc::Rc;
use wide::u64x4;
//use std::time::Instant;

fn main() -> Result<()> {
    println!("Hello from Hamming Descent");
    tracing_subscriber::fmt::init();
    // let filter = EnvFilter::from_default_env()
    //     .add_directive("debug".parse().unwrap())
    //     .add_directive("rp_forest=warn".parse().unwrap());
    // tracing_subscriber::fmt().with_env_filter(filter).init();
    //let now = Instant::now();
    tracing::info!("Loading mf dino data...");
    let num_queries = 10_000; // for runnning: 10_000;  // for testing 990_000
    let num_data = 1_000_000 - num_queries;
    let dao: Rc<Dao<Array1<f32>>> = Rc::new(dao_from_csv_dir(
        "/Volumes/Data/RUST_META/mf_dino2_csv/",
        num_data,
        num_queries,
    )?);
    let num_neighbours = 10;
    //let max_candidates = 50;

    let dao: Rc<Dao<BitVecSimd<[u64x4; 4], 4>>> = to_cubic_dao(dao.clone());

    let descent = Descent::new(
        dao.clone(),
        num_neighbours,
        true,
        hamming_distance_as_f32::<4>,
    );

    println!("First row: {:?}", descent.current_graph.nns[0]);
    println!("First row: {:?}", descent.current_graph.distances[0]);

    Ok(())
}
