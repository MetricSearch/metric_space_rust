use anyhow::Result;
use dao::csv_dao_loader::dao_from_csv_dir;
use dao::hdf5_dao_loader::hdf5_f32_write;
use dao::Dao;
use ndarray::Array1;
use std::rc::Rc;

fn main() -> Result<()> {
    tracing::info!("Loading mf dino data...");
    let num_queries = 10_000;
    let num_data = 1_000_000 - num_queries;

    let dest_data_path = "/Volumes/Data/RUST_META/mf_dino2.h5";

    let dao: Rc<Dao<Array1<f32>>> = Rc::new(dao_from_csv_dir(
        "/Volumes/Data/RUST_META/mf_dino2_csv/",
        num_data,
        num_queries,
    )?);

    hdf5_f32_write(
        dest_data_path,
        &dao.meta.name,
        &dao.meta.description,
        &dao.meta.normed.to_string(),
        &dao.embeddings,
    )?;

    Ok(())
}
