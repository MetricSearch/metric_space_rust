use std::path::{Path, PathBuf};
use std::rc::Rc;
use dao::hdf5_to_dao_loader::hdf5_f32_to_bsp_load;
use hdf5::{File as Hdf5File, File};
use bits::container::Simd256x2;
use bits::EvpBits;
use dao::Dao;

const ALL_RECORDS: usize = 0;
const NUM_VERTICES: usize = 333;
const NUM_QUERIES: usize = 0;

pub fn load_chunks( base_path: &Path, filenames: Vec<String>, required_min_size: usize ) -> anyhow::Result<Rc<Dao<EvpBits<Simd256x2, 512>>>> {
    let mut loaded = 0;
    let mut count = 0;

    let mut daos = vec![];

    while loaded < required_min_size {
        let filename = filenames.get(count).unwrap_or_else(|| { panic!("Run out of files fix me") } );

        let next_dao= Rc::new(
            hdf5_f32_to_bsp_load::<Simd256x2, 512>(
                filename,
                ALL_RECORDS,
                NUM_QUERIES,
                NUM_VERTICES,
            ).unwrap() );

        loaded = loaded + next_dao.data_len();
        daos.push(next_dao);

    }

    let one = daos.get(0).unwrap();

    Ok( one.clone() )

}

pub fn get_file_sizes(base_path: &Path, filenames: &Vec<String>) -> anyhow::Result<Vec<usize>> {

    let mut sizes = vec![];

    for filename in filenames {
        let mut path = PathBuf::from(base_path);
        path.push(filename);
        let file = File::open(path)?; // open for reading
        let h5_data = file.dataset("data")?; // the data
        sizes.push(h5_data.shape()[0] );
    }
    Ok(sizes)

}