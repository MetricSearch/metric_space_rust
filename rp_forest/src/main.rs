extern crate core;


mod tree;

use crate::tree::RPForest;
use anyhow::Result;
use dao::{Dao,dao_from_description};
use std::rc::Rc;
use std::time::Instant;
use tracing_subscriber::EnvFilter;
use dao::csv_f32_loader::csv_f32_loader;

fn main() -> Result<()> {
    let filter = EnvFilter::from_default_env().add_directive("rp_forest=warn".parse().unwrap());
    tracing_subscriber::fmt().with_env_filter(filter).init();


    //let now = Instant::now();
    tracing::info!("Loading mf dino data...");
    let num_queries = 10_000;
    let num_data = 1_000_000 - num_queries;
    let dao: Rc<Dao> = Rc::new( dao_from_description( "/Volumes/Data/RUST_META/mf_dino2_csv/meta_data.txt",num_data,num_queries ) );
    tracing::info!("mf dino data loaded, building forest...");

    let forest = RPForest::new(30, 40, dao.clone());

    tracing::info!("Forest built, querying...");

    let res = forest.lookup(dao.get(0));
    // was query not get
    println!("Number of results = {}", res.len());
    println!("{:?}", res);


    Ok(())
}
