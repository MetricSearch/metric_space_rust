
use anyhow::{anyhow, Result};
use hdf5::{File, H5Type};
use ndarray::{arr2, s, Array, Array2};
use tracing::error;

fn main() -> Result<()> {

    let data_vec: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
    // The object that we will serialize.
    let arrai  = Array::from_shape_vec((5,3), data_vec);

    // println!("Before serialisation: {:?}", arrai);
    let f_name = "../_scratch/arrai.h5";

    write_data::<f32>( f_name, &arrai.unwrap() );

    let decoded: Array2<f32> = read_data::<f32>(f_name)?;

    println!("Read from file: {:?}", decoded);
    Ok(())
}

fn write_data<T:H5Type>( fname: &str, arrai: &Array2<T> ) -> Result<()> {
    let file = File::create(fname)?;                        // open for writing
    let group = file.create_group("/embeddings")?;   // create a group
    // #[cfg(feature = "blosc")]
    // blosc_set_nthreads(2);                                      // set number of blosc threads
    let builder = group.new_dataset_builder();
    // #[cfg(feature = "blosc")]
    // let builder = builder.blosc_zstd(9, true); // zstd + shuffle
    let dim = arrai.shape()[1];
    let num_records = arrai.shape()[0];
    let ds = builder
        .with_data(arrai)
        .create("all_embeddings")?;
    // next finalize and write the dataset
    // create an attr with fixed shape but don't write the data
    let attr = ds.new_attr::<i32>().create("dim").unwrap();
    attr.write_scalar(&dim);

    let attr = ds.new_attr::<i32>().create("num_records").unwrap();
    attr.write_scalar(&num_records);
    file.flush();
    Ok(())
}

fn read_data<T:H5Type>( fname: &str ) -> Result<Array2<T>> {
    let file = File::open(fname)?; // open for reading
    let ds = file.dataset("/embeddings/all_embeddings")?; // open the dataset
    let data: Array2<T> = ds.read_slice(s![..,..] ).unwrap();      // read the dataset
    let dim_attr = ds.attr("dim")?; // open the attribute
    println!("Dim is {:?}",dim_attr.read_scalar::<usize>().unwrap());
    let num_records = ds.attr("num_records")?; // open the attribute
    println!("Num records is {:?}",num_records.read_scalar::<usize>().unwrap());

    Ok((data))
    //Err( anyhow!( "Error reading" ) )
}