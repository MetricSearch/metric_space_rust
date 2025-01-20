use itertools::{iproduct};
use rp_forest::dao::Dao;
use rp_forest::tree::RPTree;
use std::rc::Rc;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    tracing::info!("**** Loading mf dino data...");
    let num_queries = 10_000;
    let num_data = 1_000_000 - num_queries;
    let dao: Rc<Dao> = Rc::new(Dao::new(
        "/Volumes/data/mf_dino2_csv/mf_dino2.csv",
        "unused",
        num_data,
        num_queries,
    )?);
    tracing::info!("mf dino data loaded, adding data...");

    let mut tree = RPTree::new(10, dao.clone(), 7);
    for i in 0..100 {
        tree.add(i);
    }

    let binding = iproduct!(0..100, 0..100)
        .map(|(x, y)| 1. - dot_product(dao.get(x).unwrap(), dao.get(y).unwrap()))
        .collect::<Vec<f32>>();

    let rows = &binding.chunks(100).collect::<Vec<_>>();

    let rows_with_ids: Vec<Vec<(usize, &f32)>> = rows
        .iter()
        .map(|&row| row.iter().enumerate().collect::<Vec<(usize, &f32)>>())
        .collect();

    let sorted_rows_with_ids = rows_with_ids
        .iter()
        .map(|x| {
            let mut sorted_x = x.clone();
            sorted_x.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            sorted_x
        })
        .collect::<Vec<Vec<(usize, &f32)>>>();

    println!("{:?}", sorted_rows_with_ids.get(0).unwrap());
    lookup(0, dao.clone(), &mut tree)?;

    Ok(())
}

fn dot_product(p0: &[f32], p1: &[f32]) -> f32 {
    assert!(p0.len() == p1.len());
    p0.iter().zip(p1.iter()).map(|(x, y)| x * y).sum()
}

fn lookup(index: usize, dao: Rc<Dao>, tree: &mut RPTree) -> anyhow::Result<()> {
    let res = tree.lookup(dao.get(index)?);
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
