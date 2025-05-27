use std::{thread::sleep, time::Duration};

use dao::{pubmed_hdf5_mmap_loader, pubmed_hdf5_to_dao_loader::hdf5_pubmed_f32_to_bsp_load};

const PATH: &str = "/home/fm208/Downloads/pubmed/benchmark-dev-pubmed23.h5";
const DATASET: &str = "train";

fn main() -> anyhow::Result<()> {
    pretty_env_logger::formatted_timed_builder()
        .filter_level(log::LevelFilter::Trace)
        .init();

    log::info!("starting mmap");
    let test = pubmed_hdf5_mmap_loader::load(PATH, 0, 10000, 200)?;

    log::info!("starting original");
    let correct = hdf5_pubmed_f32_to_bsp_load(PATH, 0, 10000, 200)?;

    log::info!("done");

    if correct != test {
        panic!();
    } else {
        Ok(())
    }
}
