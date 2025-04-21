use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::rc::Rc;
use dao::Dao;
use anyhow::Result;
use ndarray::Array1;
use dao::csv_dao_loader::{dao_from_csv_dir};
use descent::Descent;

fn main() -> Result<()> {
    println!("Hello from serialise RNG");
    tracing_subscriber::fmt::init();

    let data_file_name = "/Volumes/Data/RUST_META/mf_dino2_csv/";
    let descent_file_name = "_scratch/nn_table_100.bin";
    let rng_star_file_name = "_scratch/rng_table_100.bin";

    println!("Loading mf dino data...");
    let num_queries = 10_000; // for runnning: 10_000;  // for testing 990_000
    let num_data = 1_000_000 - num_queries;
    let dao: Rc<Dao<Array1<f32>>> = Rc::new(dao_from_csv_dir(
        data_file_name,
        num_data,
        num_queries,
    )?);

    let f = BufReader::new(File::open(descent_file_name).unwrap());

    let descent : Descent<> = bincode::deserialize_from(f).unwrap();

    println!("Getting rng table ...");

    let rng_table = descent.rng_star(dao.clone());

    println!("Saving NN table to {} ...", rng_star_file_name);

    let f = BufWriter::new(File::create(rng_star_file_name).unwrap());
    bincode::serialize_into(f, &rng_table).unwrap();

    Ok(())
}
