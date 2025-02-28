
use std::rc::Rc;
use ndarray::Array1;
use dao::csv_f32_loader::dao_from_csv_dir;
use dao::Dao;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    tracing::info!("**** Loading mf dino data...");
    let num_queries = 0;
    let num_data = 1_000_000 - num_queries;
    let dao: Rc<Dao<Array1<f32>>> = Rc::new(dao_from_csv_dir(
        "/Volumes/Data/RUST_META/mf_dino2_csv/",
        num_data,
        num_queries,
    )?);
    tracing::info!("mf dino data loaded, getting data 0...");

    let data1 = dao.get_datum(999_999);

    tracing::info!("{:?}", data1);

    Ok(())
}
