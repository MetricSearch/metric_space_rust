use anyhow::Result;
use bits::container::Simd256x2;
use bits::EvpBits;
use clap::Parser;
use dao::{Dao, DaoMatrix, DaoMetaData};
use std::mem;
use std::sync::atomic::AtomicBool;
use utils::Nality;

#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

/// clap parser
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to HDF5
    path: String,
    //     embeddings_path: String,
}

fn main() -> Result<()> {
    pretty_env_logger::formatted_timed_builder()
        .filter_level(log::LevelFilter::Trace)
        .init();

    let dada_mem_size: i64 = 263748056 * 1024; // bytes // TODO parameter

    log::debug!("Dada memory: {} bytes", dada_mem_size);

    let dada_75_pc = dada_mem_size * 3 / 4;

    log::debug!("75% Dada memory: {} bytes", dada_75_pc);

    let num_neighbours = 18; // TODO make a parameter
    let reverse_list_size = 0; // TODO make a parameter

    let evp_size = size_of::<EvpBits<Simd256x2, 384>>();
    log::debug!("EVP size: {} bytes", evp_size);

    let dao_size = size_of::<Dao<EvpBits<Simd256x2, 384>>>() + size_of::<DaoMetaData>();

    log::debug!("Dao size: {} bytes", dao_size);

    let neighbourlarity_size = size_of::<Nality>();
    let atomic_bool_size = std::mem::size_of::<AtomicBool>();

    log::debug!("Size of single neighbourlarity {}", neighbourlarity_size);
    log::debug!("Size of single AtomicBool {}", atomic_bool_size);

    let rev_list_size_per_datum = neighbourlarity_size * reverse_list_size;
    let nn_table_entry_size_per_datum = neighbourlarity_size * num_neighbours;
    let flags_size_per_datum = atomic_bool_size * 2; // neighbour_is_new and old_flags

    // Sizes:
    // neighbourlarities: Array2<Nality> size: num_neighbours * num_rows
    // neighbour_is_new: Array2<AtomicBool>
    // old_neighbour_state: copy of neighbourlarities
    // old_flags: copy of neighbour_is_new
    // reverse_links: Array2<Nality> size: num_data, reverse_list_size

    let mem_per_datum = rev_list_size_per_datum + // reverse_links
        2 * nn_table_entry_size_per_datum + // neighbourlarities + old_neighbour_state
        2 * flags_size_per_datum; // neighbour_is_new + old_flags

    log::debug!("Structural space used per datum: {} ", mem_per_datum);

    log::debug!("Total space used per datum: {} ", mem_per_datum + evp_size);

    Ok(())
}
