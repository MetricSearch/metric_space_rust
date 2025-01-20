use rp_forest::dao::Dao;
use std::rc::Rc;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    tracing::info!("**** Loading mf dino data...");
    let num_queries = 0;
    let num_data = 1_000_000 - num_queries;
    let dao: Rc<Dao> = Rc::new(Dao::new(
        "/Volumes/data/mf_dino2_csv/mf_dino2.csv",
        "unused",
        num_data,
        num_queries,
    )?);
    tracing::info!("mf dino data loaded, getting data 0...");

    let data1 = dao.get(999_999);

    tracing::info!("{:?}", data1);

    Ok(())
}
