/*
  Program to merge NN tables from Laion-400 h5 files.
*/
use clap::Parser;

/// clap parser
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to HDF5 source
    source_path: String,
    output_path: String,
}

pub fn main() -> anyhow::Result<()> {
    //     let file_name = "nn_table".to_string().add(i.to_string().as_str());
    //     create_and_store_nn_table::<Simd256x2, 500>(
    //         dao,
    //         NUM_NEIGHBOURS,
    //         REVERSE_LIST_SIZE,
    //         UNUSED_DELETE_ME,
    //         DELTA,
    //         &args.output_path,
    //         &file_name,
    //     );
    // }

    Ok(())
}
