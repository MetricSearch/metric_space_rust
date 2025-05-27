use crate::{Dao, DaoMetaData, Normed};
use hdf5::File;
use ndarray::{s, Array1, Array2};
//use tracing::metadata;

pub fn hdf5_laion_f32_load(
    data_path: &str,
    num_records: usize,
    num_queries: usize,
) -> anyhow::Result<Dao<Array1<f32>>> {
    let file = File::open(data_path)?; // open for reading
    let ds_data = file.dataset("emb")?; // open the test dataset

    let name = "Laion-clip";
    let description = "Laion-clip-768v2-n=10M";
    let dim = 768;
    // let num_records = 10_000_000;
    // let num_queries= 1000;

    let normed = Normed::L2; //?????

    let embeddings: Array2<f32> = ds_data.read_slice(s![.., ..]).unwrap(); // read the dataset

    let embeddings = embeddings
        .rows()
        .into_iter()
        .map(|x| x.to_owned())
        .collect::<Array1<Array1<f32>>>();

    let dao_meta = DaoMetaData {
        name: name.to_string(),
        description: description.to_string(),
        data_disk_format: "".to_string(),
        path_to_data: "".to_string(),
        normed: normed,
        num_records: num_records,
        dim: dim,
    };

    let dao = Dao {
        meta: dao_meta,
        num_data: num_records - num_queries,
        num_queries: num_queries,
        embeddings: embeddings,
    };

    Ok(dao)
}
