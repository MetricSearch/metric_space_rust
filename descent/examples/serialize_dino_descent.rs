use anyhow::Result;
use dao::csv_dao_loader::dao_from_csv_dir;
use dao::Dao;
use descent::Descent;
use ndarray::Array1;
use std::fs::File;
use std::io::BufWriter;
use std::rc::Rc;
use utils::distance_f32;

fn main() -> Result<()> {
    println!("Hello from Serialize Descent example");
    tracing_subscriber::fmt::init();
    println!("Loading mf dino data...");
    let num_queries = 10_000; // for runnning: 10_000;  // for testing 990_000
    let num_data = 1_000_000 - num_queries;
    let dao: Rc<Dao<Array1<f32>>> = Rc::new(dao_from_csv_dir(
        "/Volumes/Data/RUST_META/mf_dino2_csv/",
        num_data,
        num_queries,
    )?);

    let num_neighbours = 100;
    let descent = Descent::new(dao.clone(), num_neighbours, true, distance_f32);

    println!("Saving NN table to _scratch/descent_100.bin ...");

    let f = BufWriter::new(File::create("_scratch/nn_table_100.bin").unwrap());
    let _ = bincode::serialize_into(f, &descent);

    Ok(())
}
