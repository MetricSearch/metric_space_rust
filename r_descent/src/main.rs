use dao::{dao_from_description, Dao};
use anyhow::Result;
use r_descent::descent::Descent;
use std::rc::Rc;
use tracing_subscriber::EnvFilter;
use dao::csv_f32_loader::csv_f32_loader;
//use std::time::Instant;

fn main() -> Result<()> {
    //tracing_subscriber::fmt::init();
    let filter = EnvFilter::from_default_env().add_directive("debug".parse().unwrap()).add_directive("rp_forest=warn".parse().unwrap());
    tracing_subscriber::fmt().with_env_filter(filter).init();
    //let now = Instant::now();
    tracing::info!("Loading mf dino data...");
    let num_queries = 10_000; // for runnning: 10_000;  // for testing 990_000
    let num_data = 1_000_000 - num_queries;
    let dao: Rc<Dao> = Rc::new( dao_from_description("/Volumes/Data/RUST_META/mf_dino2_csv/meta_data.txt", num_data,num_queries) );
    let num_neighbours = 10;
    //let max_candidates = 50;

    let descent = Descent::new(dao.clone(), num_neighbours, true);

    println!("First row: {:?}", descent.current_graph.indices[0]);
    println!("First row: {:?}", descent.current_graph.distances[0]);

    Ok(())
}
