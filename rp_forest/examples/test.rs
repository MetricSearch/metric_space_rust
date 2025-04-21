
use rp_forest::tree::RPTree;
use std::rc::Rc;
use ndarray::Array1;
use dao::csv_dao_loader::dao_from_csv_dir;
use dao::Dao;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    tracing::info!("**** Loading mf dino data...");
    let num_queries = 10_000;
    let num_data = 1_000_000 - num_queries;
    let dao: Rc<Dao<Array1<f32>>> = Rc::new(dao_from_csv_dir(
        "/Volumes/Data/RUST_META/mf_dino2_csv/",
        num_data,
        num_queries,
    )?);
    tracing::info!("mf dino data loaded, adding data...");

    let mut tree = RPTree::new(10, dao.clone(), 7);
    for i in 0..100 {
        //dao.data_len() {
        // if i % 100_000 == 0 {
        //     tracing::info!("Adding data {i}");
        // }
        tree.add(i);
    }

    tracing::info!("{:?}", tree);

    // lookup of 0 yields:
    // 0
    // 18
    // 44
    // 79
    // 93
    // 97 with 0..100 data items loaded

    lookup(0, dao.clone(), &mut tree)?;
    lookup(18, dao.clone(), &mut tree)?;
    lookup(44, dao.clone(), &mut tree)?;
    lookup(79, dao.clone(), &mut tree)?;
    lookup(93, dao.clone(), &mut tree)?;
    lookup(97, dao.clone(), &mut tree)?;

    Ok(())
}

fn lookup(index: usize, dao: Rc<Dao::<Array1<f32>>>, tree: &mut RPTree::<Array1<f32>>) -> anyhow::Result<()> {
    let res = tree.lookup(dao.get_datum(index).clone());
    match res {
        Some(results) => {
            println!("Number of results = {}", results.len());
            display(results);
        }
        None => {
            println!("No results found");
        }
    };
    Ok(())
}

fn display(results: Vec<usize>) {
    for result in results {
        println!("{}", result);
    }
}
