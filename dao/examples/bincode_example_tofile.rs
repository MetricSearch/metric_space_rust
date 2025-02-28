use anyhow::Result;
use ndarray::{Array, Array2};
use std::fs::File;
use std::io::{BufReader, BufWriter};

fn main() -> Result<()> {
    let data_vec: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    // The object that we will serialize.
    let arrai = Array::from_shape_vec((3, 3), data_vec);

    println!("Before serialisation: {:?}", arrai);
    let f_name = "../_scratch/arrai_data.txt";

    write_data(f_name, &arrai.unwrap())?;

    let decoded: Array2<f32> = read_data(f_name)?;

    println!("Read from file: {:?}", decoded);
    Ok(())
}

fn write_data(fname: &str, arrai: &Array2<f32>) -> Result<()> {
    let encoded = bincode::serialize(&arrai).unwrap();
    let writer: BufWriter<File> = BufWriter::new(File::create(fname)?);
    bincode::serialize_into(writer, &arrai)?;
    // writer.flush()?;
    println!("Wrote: {:?} bytes", &encoded.len());
    Ok(())
}

fn read_data(fname: &str) -> Result<Array2<f32>> {
    let f = BufReader::new(File::open(fname).unwrap());
    let decoded: Array2<f32> = bincode::deserialize_from(f).unwrap();
    Ok(decoded)
}
