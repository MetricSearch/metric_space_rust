use big_knn::write_nalities_as_json;
use big_knn::{get_row_iterator, RowIter};
use clap::Parser;
use ndarray::{Array1, Array2};
use serde::Deserialize;
use std::collections::{BTreeSet, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicU64;
use std::{cmp, fs, io};
use utils::Nality;

/// Code Merges multiple Nality files from a directory into a single file for that partition

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Directory containing .bin files
    nn_tables_source_dir: String,
    nn_tables_dest_dir: String,
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();

    pretty_env_logger::formatted_timed_builder()
        .filter_level(log::LevelFilter::Trace)
        .init();

    let input_dirpath = Path::new(&args.nn_tables_source_dir);
    // Check directory exists
    if !input_dirpath.is_dir() {
        panic!("{:?} is not a directory", args.nn_tables_source_dir);
    }
    log::trace!("Looking in  {:?}", input_dirpath);

    let output_dirpath = Path::new(&args.nn_tables_dest_dir);
    // Check directory exists
    if !output_dirpath.is_dir() {
        panic!("{:?} is not a directory", args.nn_tables_dest_dir);
    }
    log::trace!("Writing to  {:?}", output_dirpath);

    // Get all the input dirs
    let dirs: Vec<PathBuf> = fs::read_dir(input_dirpath)?
        .filter_map(|entry| {
            entry.ok().and_then(|dir_ent| {
                let path = dir_ent.path();
                if dir_ent.path().is_dir() {
                    Some(path)
                } else {
                    None
                }
            })
        })
        .collect();

    log::trace!("Found dirs: {:?}", dirs);

    for dir in dirs {
        log::trace!("Looking in dir {:?}", &dir);

        let merge_filename = dir.file_name().unwrap(); // this is the name of the dir we are in and will be used for the output file name

        // Collect files from directory
        let file_names: Vec<PathBuf> = fs::read_dir(&dir)?
            .filter_map(|entry| {
                entry.ok().and_then(|dir_ent| {
                    let path = dir_ent.path();
                    if path.is_file() && path.extension().map(|ext| ext == "json").unwrap_or(false)
                    {
                        Some(path)
                    } else {
                        None
                    }
                })
            })
            .collect();

        let mut target_name = output_dirpath.join(&merge_filename);
        target_name.set_extension("json");

        log::trace!(
            "Merging {} files: {:?} to {:?}",
            file_names.len(),
            file_names,
            target_name.to_str().unwrap()
        );

        merge_files_rowwise(&file_names, &target_name);
    }

    Ok(())
}

fn merge_files_rowwise(filenames: &Vec<PathBuf>, output_path: &PathBuf) -> io::Result<()> {
    log::trace!("Iterating over files: {:?}", filenames);

    let mut vec_of_iterators: Vec<RowIter> = filenames // iterators over each of the nn_tables that have been created already
        .iter()
        .map(|f| get_row_iterator(f.to_str().unwrap())) // TODO look at type migration str -> pathbuf -> str
        .collect();

    let mut merged_rows: Vec<Vec<Nality>> = vec![];

    let mut more_lines = true;

    while more_lines {
        // First get the lines to be merged

        let mut next_lines: Vec<Array1<Nality>> = vec![];

        for iter in vec_of_iterators.iter_mut() {
            // get each line from the file iterators
            let next_line = iter.next();
            // does the iterator still have data? - None if no data
            if let Some(next_line) = next_line {
                next_lines.push(next_line);
            } else {
                more_lines = false;
            }
        }

        // For each row do a k-way merge keeping best n
        // n determined by the size of the input rows.

        if !next_lines.is_empty() {
            let merged_row = merge_lines(&next_lines);
            merged_rows.push(merged_row);
        }
    }

    // make the vector into an ArrayVew2<Nality> as required by the write_nalities_as_json function

    let nrows = merged_rows.len();
    let ncols = merged_rows
        .first()
        .map(|a| a.len())
        .expect("Input Vec is empty"); // should never happen - all are same length an not empty.

    // Flatten into a single Vec<T>
    let flattened: Vec<Nality> = merged_rows.into_iter().flat_map(|a| a.to_vec()).collect();

    // Reshape into (nrows, ncols)
    let as_array2 = Array2::from_shape_vec((nrows, ncols), flattened)
        .expect("Shape mismatch when reshaping into Array2");

    Ok(write_nalities_as_json(output_path, &as_array2.view()))
}

fn merge_lines(lines: &Vec<Array1<Nality>>) -> Vec<Nality> {
    let length = lines.first().map(|a| a.len()).unwrap_or(0);

    let mut merged = Vec::new();
    let mut seen = BTreeSet::new();

    // Push every item from the rows into merged.

    // Uses the OrdNality Eq based on id for the set

    for line in lines {
        for nality in line.iter() {
            let ordnality = OrdNality(nality.clone());
            if !seen.contains(&ordnality) {
                merged.push(nality);
                seen.insert(ordnality);
            }
        }
    }

    // Cut merged down to the correct size (always bigger or (unlikely) the length required.
    merged.sort_by(|a, b| b.sim().partial_cmp(&a.sim()).unwrap()); // first sort the items by sim // TODO AL IS HERE <<<<<<<< LOOK AT
    merged.truncate(length);

    // strip off the borrows to return Nalities
    merged.iter().map(|&nality| nality.clone()).collect()
}

pub struct OrdNality(Nality);

// NOTE All the eqs of OrdNality are based on id only.
// must be used with extreme care due to concurrency races that may occur.

impl Eq for OrdNality {}

impl PartialEq<Self> for OrdNality {
    fn eq(&self, other: &Self) -> bool {
        self.0.id().as_usize() == other.0.id().as_usize()
    }
}

impl PartialOrd<Self> for OrdNality {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdNality {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.0.id().as_usize().cmp(&other.0.id().as_usize())
    }
}
