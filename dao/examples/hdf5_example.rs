use anyhow::Result;
use hdf5::{Dataset, File, H5Type};
use ndarray::{s, Array, Array2};
//use tracing::error;

fn main() -> Result<()> {
    let data_vec: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
    ];
    // The object that we will serialize.
    let arrai = Array::from_shape_vec((5, 3), data_vec);

    // println!("Before serialisation: {:?}", arrai);
    let f_name = "../_scratch/arrai.h5";

    let _ = write_data::<f32>(f_name, &arrai.unwrap())?;

    let decoded: Array2<f32> = read_data::<f32>(f_name)?;

    println!("Read from file: {:?}", decoded);
    Ok(())
}

fn write_data<T: H5Type>(fname: &str, arrai: &Array2<T>) -> Result<()> {
    let file = File::create(fname)?; // open for writing
    let group = file.create_group("/embeddings")?; // create a group
                                                   // #[cfg(feature = "blosc")]
                                                   // blosc_set_nthreads(2);                                      // set number of blosc threads
    let builder = group.new_dataset_builder();
    // #[cfg(feature = "blosc")]
    // let builder = builder.blosc_zstd(9, true); // zstd + shuffle
    let dim = arrai.shape()[1];
    let num_records = arrai.shape()[0];
    let ds = builder.with_data(arrai).create("all_embeddings")?;
    // next finalize and write the dataset

    add_str_attr(&ds, "name", &"test_name");
    add_str_attr(&ds, &"description", &"Mirflkr DinoV2 encoded data as f32s");
    add_attr(&ds, "dim", &dim);
    add_attr(&ds, "num_records", &num_records);
    add_str_attr(&ds, &"normed", &"L2");

    let _ = file.flush();
    Ok(())
}

fn add_attr(ds: &Dataset, key: &str, value: &usize) {
    let attr = ds.new_attr::<i32>().create(key).unwrap();
    let _ = attr.write_scalar(value);
}

pub fn add_str_attr(ds: &Dataset, key: &str, value: &str) {
    let attr = ds
        .new_attr::<hdf5::types::VarLenUnicode>()
        .create(key)
        .unwrap();
    let value_: hdf5::types::VarLenUnicode = value.parse().unwrap();
    let _ = attr.write_scalar(&value_);
}

fn read_data<T: H5Type>(fname: &str) -> Result<Array2<T>> {
    let file = File::open(fname)?; // open for reading
    let ds = file.dataset("/embeddings/all_embeddings")?; // open the dataset
    let data: Array2<T> = ds.read_slice(s![.., ..]).unwrap(); // read the dataset
    let dim_attr = ds.attr("dim")?; // open the attribute
    println!("Dim is {:?}", dim_attr.read_scalar::<usize>().unwrap());
    let num_records = ds.attr("num_records")?; // open the attribute
    println!(
        "Num records is {:?}",
        num_records.read_scalar::<usize>().unwrap()
    );

    Ok(data)
}
