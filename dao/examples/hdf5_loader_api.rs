
use anyhow::{anyhow, Result};
use hdf5::{File, H5Type};
use ndarray::{arr2, s, Array, Array2};
use dao::hdf5_f32_loader::{hdf5_f32_load, hdf5_f32_write};

fn main() -> Result<()> {

    let data_vec: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
    // The object that we will serialize.
    let arrai  = Array::from_shape_vec((5,3), data_vec);

    // println!("Before serialisation: {:?}", arrai);
    let f_name = "../_scratch/arrai.h5";

    hdf5_f32_write( f_name, &arrai.unwrap() );

    let decoded: Array2<f32> = hdf5_f32_load(f_name)?;

    println!("Read from file: {:?}", decoded);
    Ok(())
}
