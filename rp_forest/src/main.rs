extern crate core;

mod dao;
mod tree;

use crate::tree::RPForest;
use anyhow::Result;
use dao::Dao;
use std::rc::Rc;
use std::time::Instant;

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    //let now = Instant::now();
    tracing::info!("Loading mf dino data...");
    let num_queries = 10_000;
    let num_data = 1_000_000 - num_queries;
    let dao: Rc<Dao> = Rc::new(Dao::new(
        "/Volumes/data/mf_dino2_csv/mf_dino2.csv",
        "unused",
        num_data,
        num_queries,
    )?);
    tracing::info!("mf dino data loaded, building forest...");

    let forest = RPForest::new(30, 40, dao.clone());

    tracing::info!("Forest built, querying...");

    if let res = forest.lookup(dao.get(0)?) {
        // was query not get
        println!("Number of results = {}", res.len());
        println!("{:?}", res);
    }

    Ok(())
}
