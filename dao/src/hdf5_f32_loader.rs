use std::io::Write;
use anyhow::anyhow;
use ndarray::{s, Array2};
use hdf5::{File, H5Type};

pub fn hdf5_f32_load(data_path: &str) -> anyhow::Result<Array2<f32>> {
    let file = File::open(data_path)?; // open for reading
    let ds = file.dataset("/embeddings/all_embeddings")?; // open the dataset
    let dim = ds.attr("dim")?; // open the attribute
    let num_records = ds.attr("num_records")?; // open the attribute
    let data: Array2<f32> = ds.read_slice(s![..,..] ).unwrap();      // read the dataset
    Ok(data)
}

pub fn hdf5_f32_write<T:H5Type>( data_path: &str, arrai: &Array2<T> ) -> anyhow::Result<()> {
    let file = File::create(data_path)?;                        // open for writing
    let group = file.create_group("/embeddings")?;   // create a group
    // A blocking, shuffling and loss-less compression
    // #[cfg(feature = "blosc")]
    // blosc_set_nthreads(2);                                      // set number of blosc threads
    let builder = group.new_dataset_builder();
    //#[cfg(feature = "blosc")]
    //let builder = builder.blosc_zstd(9, true); // zstd + shuffle
    let dim = arrai.shape()[1];
    let num_records = arrai.shape()[0];
    let ds = builder
        .with_data(arrai)
        .create("all_embeddings")?;
    // next finalize and write the dataset
    let attr = ds.new_attr::<i32>().create("dim").unwrap();
    attr.write_scalar(&dim);
    let attr = ds.new_attr::<i32>().create("num_records").unwrap();
    attr.write_scalar(&num_records);
    file.flush();
    Ok(())
}
