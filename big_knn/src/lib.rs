#![feature(path_add_extension)]

mod dao_manager;
mod knn_r_descent;
mod table_initialisation;

use crate::knn_r_descent::into_big_knn_r_descent;
use anyhow::anyhow;
use bincode;
use bits::container::{BitsContainer, Simd256x2};
use bits::EvpBits;
use dao::hdf5_to_dao_loader::hdf5_f32_to_bsp_load;
use dao::Dao;
use hdf5::File as Hdf5File;
use ndarray::ArrayView2;
use std::fs;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::rc::Rc;

const ALL_RECORDS: usize = 0;
const NUM_QUERIES: usize = 0;

pub fn save_to_h5(f_name: &str, data: ArrayView2<usize>) -> anyhow::Result<()> {
    let file = Hdf5File::create(f_name)?; // open for writing
    let group = file.create_group("/knns")?; // create a group
                                             // TODO do they need the dists too?
    let builder = group.new_dataset_builder();

    let _ds = builder.with_data(&data.to_owned()).create("results")?;

    file.flush()?;

    println!("NN table saved to bin file");

    Ok(())
}

/// Partitions up a bunch of h5 files from the base_path directory
/// into groups of files containing at most max_entries rows
pub fn get_partitions<'a>(
    dir_base: &'a Path,
    max_entries: usize,
) -> (Vec<usize>, Vec<Vec<String>>) {
    let file_names = get_file_names(dir_base).unwrap();
    let sizes = get_file_sizes(dir_base, &file_names).unwrap();
    let partition_boundaries = partition_into(&sizes, max_entries).unwrap();
    let partitions = make_partitions(&partition_boundaries, &file_names);
    (sizes, partitions)
}

pub fn get_file_names<P: AsRef<Path>>(path: P) -> anyhow::Result<Vec<String>> {
    let mut file_names = Vec::new();

    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            if let Some(name) = path.file_name() {
                if let Some(name_str) = name.to_str() {
                    file_names.push(name_str.to_string());
                }
            }
        }
    }

    file_names.sort_by(|a, b| {
        extract_index(a)
            .unwrap_or(0)
            .cmp(&extract_index(b).unwrap_or(0))
    });

    Ok(file_names)
}

// extract the numeric index from a filename.
fn extract_index(filename: &str) -> Option<u32> {
    // Extract the number between "img_emb_" and ".h5"
    filename
        .strip_prefix("img_emb_")?
        .strip_suffix(".h5")?
        .parse::<u32>()
        .ok()
}

/// Finds the sizes of all the h5 files in the directory base_path whose names are in the filenames vector
/// returns a Vector of the same size as the filenames Vector containing the number of entries in the h5 file.
/// Fails if any of the files are not h5 files
pub fn get_file_sizes(base_path: &Path, filenames: &Vec<String>) -> anyhow::Result<Vec<usize>> {
    let mut sizes = vec![];

    for filename in filenames {
        let mut path = PathBuf::from(base_path);
        path.push(filename);
        let file = Hdf5File::open(path)?; // open for reading
        let h5_data = file.dataset("data")?; // the data
        sizes.push(h5_data.shape()[0]);
    }
    Ok(sizes)
}

// Partitions the h5 files into blocks of at most max_entries_per_block and returns the starting index of weach block
pub fn partition_into(
    sizes: &Vec<usize>,
    max_entries_per_block: usize,
) -> anyhow::Result<Vec<usize>> {
    let mut partitions = vec![];

    let mut entries_accumulated = 0;

    for index in 0..sizes.len() {
        if sizes[index] > max_entries_per_block {
            return Err(anyhow!(
                "Block size at index {} too large - size: {}",
                index,
                sizes[index]
            ));
        }
        if entries_accumulated + sizes[index] > max_entries_per_block {
            //  block filled
            partitions.push(index); // so start a new block
            entries_accumulated = sizes[index]; // reset the counter
        } else {
            entries_accumulated += sizes[index]; // otherwise increment the counter
        }
    }
    if entries_accumulated > 0 {
        // push the last index if there is data accumulated
        partitions.push(sizes.len());
    }

    Ok(partitions)
}

// Takes some filename partitions and returns a Vec of Vecs containing the files partitioned up
// each outer Vec entry contains a pair containing the partition size along with a Vec of filenames from the partition.
pub fn make_partitions<'a>(
    partition_boundaries: &Vec<usize>,
    file_names: &'a Vec<String>,
) -> Vec<Vec<String>> {
    let mut result = vec![];

    let mut start: usize = 0;
    for i in 0..partition_boundaries.len() {
        let mut files_in_partition = vec![];
        for j in start..partition_boundaries[i] {
            files_in_partition.push(file_names[j].clone());
        }
        start = partition_boundaries[i];
        result.push(files_in_partition);
    }
    result
}

pub fn create_and_store_nn_table(
    dao: Dao<EvpBits<Simd256x2, 512>>,
    num_neighbours: usize,
    reverse_list_size: usize,
    delta: f64,
    start_index: usize,
    output_dir: &String,
    output_file: &String,
) {
    let vec_dao: Vec<Dao<EvpBits<Simd256x2, 512>>> = vec![dao];

    //let descent = vec_dao.into_big_knn_r_descent(num_neighbours, reverse_list_size, delta);
    let descent = into_big_knn_r_descent(vec_dao, num_neighbours, reverse_list_size, delta);

    let output_path = Path::new(&output_dir)
        .join(output_file)
        .with_added_extension("bin");

    log::info!("Writing to bin file {}", output_path.to_str().unwrap());

    let file: File = File::create(output_path).unwrap();
    let writer = BufWriter::new(file);
    let _ = bincode::serialize_into(writer, &descent);

    println!("NN table saved to bin file");
}
