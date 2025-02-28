use anyhow::Result;
use dao::hdf5_f32_loader::{hdf5_f32_load, hdf5_f32_write};
use dao::Dao;
use ndarray::{array, Array1};

fn main() -> Result<()> {
    let arrai: Array1<Array1<f32>> = array![
        array![1.0, 2.0, 3.0],
        array![4.0, 5.0, 6.0],
        array![7.0, 8.0, 9.0],
        array![10.0, 11.0, 12.0],
        array![13.0, 14.0, 15.0]
    ];

    // println!("Before serialisation: {:?}", arrai);
    let f_name = "../_scratch/arrai.h5";

    hdf5_f32_write(
        f_name,
        &"Benny the Ball".to_string(),
        &"fish".to_string(),
        &"L2".to_string(),
        &arrai,
    )?;

    let decoded = hdf5_f32_load(f_name, 999_000, 10_000)?;

    println!("Read from file: {:?}", decoded.embeddings);
    Ok(())
}

pub fn dao_from_h5(
    data_path: &str,
    num_data: usize,
    num_queries: usize,
) -> Result<Dao<Array1<f32>>> {
    hdf5_f32_load(data_path, num_data, num_queries)
}
