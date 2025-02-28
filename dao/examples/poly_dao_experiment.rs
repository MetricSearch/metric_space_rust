use bitvec_simd::BitVecSimd;
use hdf5::{Dataset, File};
use ndarray::{s, Array1, Array2, ArrayBase, ArrayView1, Ix1, OwnedRepr};
use wide::u64x4;

pub trait Dao<DataRep> {
    fn get_dim(&self) -> usize;

    fn data_len(&self) -> usize;

    fn query_len(&self) -> usize;

    fn get_datum(&self, id: usize) -> DataRep;

    fn get_query(&self, id: usize) -> DataRep;

    fn get_data(&self) -> ArrayView1<DataRep>;

    fn get_queries(&self) -> ArrayView1<DataRep>;
}

pub struct DaoStruct<DataRep> {
    dim: usize,
    num_data: usize,
    num_queries: usize,
    embeddings: Array1<DataRep>,
}

impl<T: Clone> Dao<T> for DaoStruct<T> {
    // for dao_struct<Array1<f32>> {
    fn get_dim(&self) -> usize {
        self.dim
    }

    fn data_len(&self) -> usize {
        self.num_data
    }

    fn query_len(&self) -> usize {
        self.num_queries
    }

    fn get_datum(&self, id: usize) -> T {
        if id >= self.num_data {
            panic!("id out of bounds");
        }
        self.embeddings.get(id).unwrap().clone() // copying
    }

    fn get_query(&self, id: usize) -> T {
        if id < self.num_data && id >= (self.num_queries + self.num_data) {
            panic!("id out of bounds");
        }
        self.embeddings.get(id).unwrap().clone() // copying
    }

    fn get_data(&self) -> ArrayView1<T> {
        let data = self.embeddings.slice(s![0..self.num_data]);
        data
    }

    fn get_queries(&self) -> ArrayView1<T> {
        let queries = self.embeddings.slice(s![self.num_queries..]);
        queries
    }
}

fn main() {
    let s: DaoStruct<Array1<f32>> = DaoStruct {
        dim: 2,
        num_data: 2,
        num_queries: 1,
        embeddings: Array1::from_iter((0..5).map(|_x| Array1::from_vec(vec![1.2, 2.0]))),
    };

    let t: DaoStruct<BitVecSimd<[u64x4; 4], 4>> = DaoStruct {
        dim: 2,
        num_data: 2,
        num_queries: 1,
        embeddings: Array1::from_iter(
            (0..5).map(|_| embedding_to_bitrep(Array1::from_vec(vec![1.0, -1.2, 2.0]))),
        ),
    };

    println!("{:?}", s.get_query(0));

    println!("{:?}", t.get_datum(0));
}

pub fn embedding_to_bitrep(
    embedding: ArrayBase<OwnedRepr<f64>, Ix1>,
) -> BitVecSimd<[wide::u64x4; 4], 4> {
    BitVecSimd::from_bool_iterator(embedding.iter().map(|&x| x < 0.0))
}

pub fn hdf5_f32_load(
    data_path: &str,
    num_data: usize,
    num_queries: usize,
) -> anyhow::Result<DaoStruct<Array1<f32>>> {
    let file = File::open(data_path)?; // open for reading
    let ds: Dataset = file.dataset("/embeddings/all_embeddings")?; // open the dataset

    let _ = read_string_attr(&ds, "name");
    let _: String = read_string_attr(&ds, "description");
    let dim = read_scalar_attr(&ds, "dim");
    let _ = read_scalar_attr(&ds, "num_records");

    let data: Array2<f32> = ds.read_slice(s![.., ..]).unwrap(); // read in the 2D structure

    // let data : Array1<Array1<f32>> = data.into_shape( (num_data,dim) ).unwrap();

    let data = data
        .rows()
        .into_iter()
        .map(|x| x.to_owned())
        .collect::<Array1<Array1<f32>>>();

    // let dao_meta = DaoMetaData{
    //     name: name,
    //     description: description,
    //     data_disk_format: "".to_string(),
    //     path_to_data: "".to_string(),
    //     normed: normed,
    //     num_records: num_records,
    //     dim: dim };

    let dao = DaoStruct::<Array1<f32>> {
        dim: dim,
        num_data: num_data,
        num_queries: num_queries,
        embeddings: data,
    };

    Ok(dao)
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
