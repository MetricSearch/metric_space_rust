use crate::{Dao, DaoMetaData, Normed};
use hdf5::{Dataset, File};
use ndarray::{s, Array1, Array2};
//use tracing::metadata;

pub fn hdf5_f32_load(
    data_path: &str,
    num_data: usize,
    num_queries: usize,
) -> anyhow::Result<Dao<Array1<f32>>> {
    let file = File::open(data_path)?; // open for reading
    let ds = file.dataset("/embeddings/all_embeddings")?; // open the dataset

    let name = read_string_attr(&ds, "name");
    let description: String = read_string_attr(&ds, "description");
    let dim = read_scalar_attr(&ds, "dim");
    let num_records = read_scalar_attr(&ds, "num_records");

    let normed_str: String = read_string_attr(&ds, "normed");
    let normed = match normed_str.as_str() {
        "L1" => Normed::L1,
        "L2" => Normed::L2,
        _ => Normed::None,
    };

    let data: Array2<f32> = ds.read_slice(s![.., ..]).unwrap(); // read the dataset

    let data = data
        .rows()
        .into_iter()
        .map(|x| x.to_owned())
        .collect::<Array1<Array1<f32>>>();

    let dao_meta = DaoMetaData {
        name: name,
        description: description,
        data_disk_format: "".to_string(),
        path_to_data: "".to_string(),
        normed,
        num_records,
        dim,
    };

    let dao = Dao {
        meta: dao_meta,
        num_data,
        base_addr: 0,
        num_queries,
        embeddings: data,
    };

    Ok(dao)
}

pub fn hdf5_f32_write(
    data_path: &str,
    name: &str,
    description: &str,
    normed: &str,
    arrai: &Array1<Array1<f32>>,
) -> anyhow::Result<()> {
    let file = File::create(data_path)?; // open for writing
    let group = file.create_group("/embeddings")?; // create a group
                                                   // A blocking, shuffling and loss-less compression
                                                   // #[cfg(feature = "blosc")]
                                                   // blosc_set_nthreads(2);                                      // set number of blosc threads
    let builder = group.new_dataset_builder();
    //#[cfg(feature = "blosc")]
    //let builder = builder.blosc_zstd(9, true); // zstd + shuffle
    let dim = arrai.len();
    let num_records = arrai[0].len();

    let arrai: Array1<f32> = arrai
        .iter()
        .map(|row| row.iter()) // 1D array of f32s
        .flatten()
        .map(|x| *x)
        .collect::<Array1<f32>>();

    let arrai: Array2<f32> = arrai
        .into_shape_with_order([num_records, dim])
        .unwrap()
        .into();

    let ds = builder.with_data(&arrai).create("all_embeddings")?;

    add_str_attr(&ds, "name", name);
    add_str_attr(&ds, &"description", description);
    add_attr(&ds, "dim", &dim);
    add_attr(&ds, "num_records", &num_records);
    add_str_attr(&ds, &"normed", normed);

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

pub fn read_scalar_attr(ds: &Dataset, key: &str) -> usize {
    let attr = ds.attr(key).unwrap(); // open the attribute
    attr.read_scalar::<usize>().unwrap()
}

pub fn read_string_attr(ds: &Dataset, key: &str) -> String {
    let attr = ds.attr(key).unwrap(); // open the attribute
    let varlen_str = attr.read_raw::<hdf5::types::VarLenUnicode>().unwrap();
    varlen_str.iter().map(|x| x.to_string()).collect()
}
