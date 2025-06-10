use clap::Parser;
use hdf5::File;
use ndarray::s;

/// clap parser
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to input HDF5
    input_path: String,
    /// Path to results file
    results_path: String,
}

fn main() -> anyhow::Result<()> {
    pretty_env_logger::formatted_timed_builder()
        .filter_level(log::LevelFilter::Trace)
        .init();

    let args = Args::parse();

    let input = File::open(args.input_path)?
        .group("otest")?
        .dataset("knns")?;

    let result = File::open(args.results_path)?
        .group("results")?
        .dataset("results")?;

    //  assert_eq!(input.shape()[0], result.shape()[0]);

    let mut hit = 0;
    let mut missed = 0;

    for row in 0..result.shape()[0] {
        let input_row = input.read_slice_1d::<usize, _>(s![row, ..30])?;
        let result_row = result.read_slice_1d::<usize, _>(s![row, ..30])?;

        for element in result_row {
            if input_row.iter().any(|e| *e == element) {
                hit += 1;
            } else {
                missed += 1;
            }
        }
    }

    let percent = ((hit as f64) * 100.) / ((missed + hit) as f64);

    println!("{percent:.2} ({hit}:{missed})");

    Ok(())
}
