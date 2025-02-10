use anyhow::Result;
use dao::csv_f32_loader::{dao_from_csv_dir};
use dao::Dao;
use r_descent::descent::Descent;
use std::rc::Rc;
use ndarray::Array1;
use tracing_subscriber::EnvFilter;
//use std::time::Instant;

fn main() -> Result<()> {
    //tracing_subscriber::fmt::init();
    let filter = EnvFilter::from_default_env()
        .add_directive("debug".parse().unwrap())
        .add_directive("rp_forest=warn".parse().unwrap());
    tracing_subscriber::fmt().with_env_filter(filter).init();
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

    let descent = Descent::new(dao.clone(), num_neighbours, true);

    println!("First row: {:?}", descent.current_graph.indices[0]);
    println!("First row: {:?}", descent.current_graph.distances[0]);

    Ok(())
}
