use std::io::Write;
use anyhow::anyhow;
use ndarray::{s, Array, Array2};
use hdf5::{Dataset, File, H5Type};
use crate::{Dao32, DaoMetaData, Normed};

pub fn hdf5_f32_load(data_path: &str, num_data: usize, num_queries: usize) -> anyhow::Result<Dao32> {
    let file = File::open(data_path)?; // open for reading
    let ds = file.dataset("/embeddings/all_embeddings")?; // open the dataset

    let name = read_string_attr(&ds,"name");
    let description: String = read_string_attr(&ds,"description");
    let dim = read_scalar_attr(&ds,"dim");
    let num_records = read_scalar_attr(&ds,"num_records");

    let normed_str: String = read_string_attr(&ds,"normed");
    let normed = match normed_str.as_str() {
        "L1" => Normed::L1,
        "L2" => Normed::L2,
        _ => Normed::None,
    };

    let data: Array2<f32> = ds.read_slice(s![..,..] ).unwrap();      // read the dataset

    let dao_meta = DaoMetaData{
        name: name,
        description: description,
        data_disk_format: "".to_string(),
        path_to_data: "".to_string(),
        normed: normed,
        num_records: num_records,
        dim: dim };

    let dao = Dao32::new(dao_meta, data, num_data, num_queries);

    Ok(dao)
}

pub fn hdf5_f32_write<T:H5Type>( data_path: &str, name: &str, description: &str, normed: &str, arrai: &Array2<T> ) -> anyhow::Result<()> {
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

    add_str_attr(&ds,"name", name);
    add_str_attr(&ds,&"description", description );
    add_attr(&ds, "dim", &dim);
    add_attr(&ds, "num_records", &num_records);
    add_str_attr(&ds,&"normed", normed );

    file.flush();
    Ok(())
}

fn add_attr(ds: &Dataset, key: &str, value: &usize) {
    let attr = ds.new_attr::<i32>().create(key).unwrap();
    attr.write_scalar(value);
}

pub fn add_str_attr(ds: &Dataset, key: &str, value: &str) {
    let attr = ds.new_attr::<hdf5::types::VarLenUnicode>().create(key).unwrap();
    let value_: hdf5::types::VarLenUnicode = value.parse().unwrap();
    attr.write_scalar(&value_);
}

pub fn read_scalar_attr(ds: &Dataset, key: &str) -> usize {
    let attr = ds.attr(key).unwrap(); // open the attribute
    attr.read_scalar::<usize>().unwrap()
}

pub fn read_string_attr(ds: &Dataset, key: &str) -> String {
    let attr = ds.attr(key).unwrap(); // open the attribute
    let varlen_str = attr.read_raw::<hdf5::types::VarLenUnicode>().unwrap();
    varlen_str.iter().map(|x| x.to_string()).collect()
}

