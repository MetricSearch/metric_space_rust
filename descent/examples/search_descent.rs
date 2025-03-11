use std::rc::Rc;
use dao::Dao;
use anyhow::Result;
use ndarray::Array1;
use dao::csv_f32_loader::{dao_from_csv_dir};
use descent::Descent;

fn main() -> Result<()> {
    println!("Hello from Descent RNG* example");
    tracing_subscriber::fmt::init();
    println!("Loading mf dino data...");
    let num_queries = 10_000; // for runnning: 10_000;  // for testing 990_000
    let num_data = 1_000_000 - num_queries;
    let dao: Rc<Dao<Array1<f32>>> = Rc::new(dao_from_csv_dir(
        "/Volumes/Data/RUST_META/mf_dino2_csv/",
        num_data,
        num_queries,
    )?);

    let num_neighbours = 10;

    let descent = Descent::new(dao.clone(), num_neighbours, true);

    //let serialised = serde_bincode::to_string(&descent)?;

    println!("Getting rng table ...");

    let rng_table = descent.rng_star(dao.clone());

    println!("Querying ...");

    let results = descent.knn_search(dao.get_query(0).to_owned(), rng_table, dao.clone(), 100 );

    println!("Results: {:?}", results);

    Ok(())
}
