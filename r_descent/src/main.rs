use dao::Dao;
use anyhow::Result;
use r_descent::descent::Descent;
use std::rc::Rc;
use dao::csv_loader::csv_loader;
//use std::time::Instant;

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    //let now = Instant::now();
    tracing::info!("Loading mf dino data...");
    let num_queries = 10_000;
    let num_data = 1_000_000 - num_queries;
    let dao: Rc<Dao> = Rc::new(Dao::new("/Volumes/data/mf_dino2_csv/mf_dino2.csv", num_data, num_queries, &csv_loader)?);
    let num_neighbours = 10;
    //let max_candidates = 50;

    let descent = Descent::new(dao.clone(), num_neighbours);

    println!("First row: {:?}", descent.current_graph.indices[0]);

    Ok(())
}
