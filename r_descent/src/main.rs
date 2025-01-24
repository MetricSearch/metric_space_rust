use dao::Dao;
use anyhow::Result;
use r_descent::descent::Descent;
use std::rc::Rc;
use tracing_subscriber::EnvFilter;
use dao::csv_loader::csv_loader;
//use std::time::Instant;

fn main() -> Result<()> {
    //tracing_subscriber::fmt::init();
    let filter = EnvFilter::from_default_env().add_directive("debug".parse().unwrap()).add_directive("rp_forest=warn".parse().unwrap());
    tracing_subscriber::fmt().with_env_filter(filter).init();
    //let now = Instant::now();
    tracing::info!("Loading mf dino data...");
    let num_queries = 10_000; // for runnning: 10_000;  // for testing 990_000
    let num_data = 1_000_000 - num_queries;
    let dao: Rc<Dao> = Rc::new(Dao::new("/Volumes/data/mf_dino2_csv/mf_dino2.csv", num_data, num_queries, &csv_loader)?);
    let num_neighbours = 10;
    //let max_candidates = 50;

    let descent = Descent::new(dao.clone(), num_neighbours, true);

    println!("First row: {:?}", descent.current_graph.indices[0]);
    println!("First row: {:?}", descent.current_graph.distances[0]);

    Ok(())
}
