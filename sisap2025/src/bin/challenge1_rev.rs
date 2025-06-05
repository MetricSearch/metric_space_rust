// /*
// This task challenges participants to develop memory-efficient indexing solutions with reranking capabilities. Each solution will be run in a Linux container with limited memory and storage resources.
// Container specifications: 8 virtual CPUs, 16 GB of RAM, the dataset will be mounted read-only into the container.
// Wall clock time for the entire task: 12 hours.
// Minimum average recall to be considered in the final ranking: 0.7.
// Dataset: PUBMED23 (23 million vectors (384 dimensions) with out-of-distribution queries).
// The goal is to evaluate k=30 nearest neighbors for a large set of query objects, as follows:
// The final score of each team is measured as the best throughput evaluated on up to 16 different search hyperparameters.
// Teams are provided with a public set of 11,000 query objects for development purposes.
// A private set of 10,000 new queries will be used for the final evaluation.
//  */
// // Code originates from metric_space/r_descent/examples/check_r_descent_bsp_pubmed.rs

#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

use anyhow::Result;
use bits::{bsp_distance_as_f32, EvpBits};
use clap::Parser;
use dao::csv_dao_loader::dao_from_csv_dir;
use dao::hdf5_to_dao_loader::hdf5_f32_to_bsp_load;
use dao::pubmed_hdf5_gt_loader::hdf5_pubmed_gt_load;
use dao::Dao;
use ndarray::{s, Array1, Array2, ArrayView1};
use r_descent::{
    IntoRDescent, IntoRDescentWithRevNNs, KnnSearch, RDescent, RDescentWithRev, RevSearch,
};
use std::rc::Rc;
use std::time::Instant;
use utils::ndcg;
use utils::pair::Pair;

/// clap parser
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to HDF5
    path: String,
}

fn main() -> Result<()> {
    pretty_env_logger::formatted_timed_builder()
        .filter_level(log::LevelFilter::Trace)
        .init();

    let args = Args::parse();

    log::info!("Loading Pubmed data...");
    let start = Instant::now();

    let num_queries = 10_000;
    const ALL_RECORDS: usize = 0;
    const NUM_VERTICES: usize = 256;

    let dao_bsp: Rc<Dao<EvpBits<2>>> =
        Rc::new(hdf5_f32_to_bsp_load(&args.path, ALL_RECORDS, num_queries, NUM_VERTICES).unwrap());

    let queries: ArrayView1<EvpBits<2>> = dao_bsp.get_queries();

    let queries = queries.slice(s!(0..1000));

    let data: ArrayView1<EvpBits<2>> = dao_bsp.get_data();

    log::info!(
        "Pubmed data size: {} queries size: {}, num data: {}",
        data.len(),
        queries.len(),
        dao_bsp.num_data,
    );

    let start_post_load = Instant::now();

    let num_neighbours = 8;
    let chunk_size = 200;
    let delta = 0.01;
    let reverse_list_size = 20;
    let num_reverse_neighbours: usize = 16;

    log::info!("Getting NN table");

    let descent = dao_bsp.clone().into_rdescent_with_rev_nn(
        num_neighbours,
        reverse_list_size,
        chunk_size,
        delta,
        num_reverse_neighbours,
    );

    let end = Instant::now();

    let neighbours = &descent.rdescent.neighbours;
    let rev_neighbours = &descent.reverse_neighbours;

    // println!("====== Printing First 10 Rows of neighbours ======");
    // for i in 0..1000 {
    //     println!(
    //         "{:?}",
    //         neighbours
    //             .row(i)
    //             .slice(s![0..])
    //             .iter()
    //             .map(|x| x + 1)
    //             .collect::<Vec<usize>>()
    //     );
    // }

    // println!("====== Printing First 1000 Rows of rev_neighbours ======");
    // for i in 0..10 {
    //     println!(
    //         "{:?}",
    //         rev_neighbours
    //             .row(i)
    //             .slice(s![0..])
    //             .iter()
    //             .map(|x| x + 1)
    //             .collect::<Vec<usize>>()
    //     );
    // }

    log::info!(
        "Finished (including load time in {} s",
        (end - start).as_secs()
    );

    let knns = 30;

    // let (gt_nns, _gt_dists) = hdf5_pubmed_gt_load(&args.path, knns).unwrap(); // NO POINT WITH GOOAK
    // let gt_queries = dao_bsp.get_queries();

    log::info!("Pubmed Results:");

    println!("Doing {:?} queries", queries.len());

    do_queries(
        queries.to_vec(),
        &descent,
        dao_bsp.clone(),
        //&gt_nns,
        bsp_distance_as_f32,
    );

    Ok(())
}

fn do_queries(
    queries: Vec<EvpBits<2>>,
    descent: &RDescentWithRev,
    dao: Rc<Dao<EvpBits<2>>>,
    //gt_nns: &Array2<usize>,
    distance: fn(&EvpBits<2>, &EvpBits<2>) -> f32,
) {
    queries.iter().enumerate().for_each(|(qid, query)| {
        let now = Instant::now();
        let (dists, qresults) = descent.rev_search(query.clone(), dao.clone(), 100, distance);
        let (dists, qresults) = ADD_ONE_TO_RESULTS(dists, qresults);
        let after = Instant::now();
        print!(
            "Results for Q{}\tTime per query\t{} ns\tFirst 10\t",
            qid,
            (after - now).as_nanos()
        );
        show_results(qid, &qresults);
        //show_gt(qid, gt_nns.row(qid));
        //println!("Number of GT results = {} ", gt_nns.row(qid).len());
        // println!(
        //     "Intersection size: {}",
        //     intersection_size(&qresults, gt_nns.row(qid))
        //);
    });
}

fn show_results(qid: usize, results: &Vec<Pair>) {
    results.iter().by_ref().take(10).for_each(|pair| {
        print!("{}\td\t{}\t", pair.index, pair.distance.as_f32());
    });
    println!();
}

fn show_gt(qid: usize, gt_data: ArrayView1<usize>) {
    print!(
        "GT pairs size {} first few GT results for q{}:\t",
        gt_data.len(),
        qid
    );
    gt_data.iter().take(5).for_each(|index| {
        print!("{}", index);
    });
    println!();
}

fn intersection_size(results: &Vec<Pair>, gt_indices: ArrayView1<usize>) -> usize {
    results
        .iter()
        .filter_map(|pair| {
            if gt_indices.as_slice().unwrap().contains(&pair.index) {
                Some(1)
            } else {
                None
            }
        })
        .count()
}

fn ADD_ONE_TO_RESULTS(length: usize, results: Vec<Pair>) -> (usize, Vec<Pair>) {
    let adjusted_results = results
        .into_iter()
        .map(|pair| Pair::new(pair.distance, pair.index + 1))
        .collect();

    (length, adjusted_results)
}
