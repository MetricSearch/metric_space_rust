use anyhow::Result;
use bits::{bsp_distance_as_f32, f32_data_to_cubic_bitrep, whamming_distance, Bsp};
use bitvec_simd::BitVecSimd;
use metrics::euc;
use ndarray::{Array1, Array2, ArrayView1};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::rc::Rc;
use std::time::Instant;
use serde::__private::de::borrow_cow_bytes;
use dao::{Dao};
use dao::convert_f32_to_cubic::to_cubic_dao;
use dao::csv_dao_loader::dao_from_csv_dir;
use dao::pubmed_hdf5_gt_loader::hdf5_pubmed_gt_load;
use dao::pubmed_hdf5_to_dao_loader::hdf5_pubmed_f32_to_bsp_load;
use utils::{arg_sort_2d, ndcg};
use utils::non_nan::NonNan;
use r_descent_matrix::{KnnSearch, RDescentMatrix};
use utils::pair::Pair;
//use divan::Bencher;

fn main() -> Result<()> {
    tracing::info!("Loading Pubmed data...");
    let num_queries = 10_000;

    let f_name = "/Volumes/Data/sisap_challenge_25/pubmed/benchmark-dev-pubmed23.h5";

    let descent_file_name = "_scratch/pubmed_table_10.bin";

    println!("Serde load of Pubmed data");
    let f = BufReader::new(File::open(descent_file_name).unwrap());
    let descent: RDescentMatrix = bincode::deserialize_from(f).unwrap();

    println!("Loading pubmed data...");
    let num_queries = 10_000;
    const NUM_DATA: usize = 0;
    const NUM_VERTICES: usize = 200;
    const knns: usize = 30;

    let dao_bsp: Rc<Dao<Bsp<2>>> = Rc::new(hdf5_pubmed_f32_to_bsp_load( f_name, NUM_DATA , num_queries, NUM_VERTICES ).unwrap());

    println!( "Dao: size: {} data shape: {:?} queries shape: {:?} ", dao_bsp.num_data, dao_bsp.get_data().shape(),dao_bsp.get_queries().shape() );

    println!("Loading pubmed GT data...");

    let (gt_nns,gt_dists) = hdf5_pubmed_gt_load(f_name,knns).unwrap();

    let gt_pairs = gt_nns.rows()
        .into_iter()
        .zip(gt_dists.rows())
        .map( |(idx_row, dist_row)| {
            let mut pairs =  idx_row.into_iter()
                .zip(dist_row)
                .map( |(i, d)| Pair::new( NonNan(*d),*i ) )
            .collect::<Vec<Pair>>();
            pairs.sort(); // Pair has Ord _by( |a, b| { a.distance.0.cmp(  b.distance.0 ) } );
            pairs
        } )
        .collect::<Vec<Vec<Pair>>>();

    println!("Running queries");

    let queries = dao_bsp.get_queries().to_vec();

    let this_many = 10;

    let (queries, _rest) = queries.split_at(this_many);

    let nn_table = &descent.indices;

    println!("Doing {:?} queries", queries.len());

    do_queries(queries.to_vec(),&descent,dao_bsp.clone(),&gt_pairs,nn_table, bsp_distance_as_f32);

    Ok(())
}


fn show_results(qid : usize, results: &Vec<Pair>) {
    print!( "first few results for q{}:\t", qid );
    results
        .iter()
        .by_ref()
        .take(5)
        .for_each(|pair| { print!("{} d: {} ", pair.index, pair.distance.0 ); });
    println!();
}

fn show_gt(qid : usize, gt_pairs: &Vec<Vec<Pair>>) {
    print!( "GT pairs size {} first few GT results for q{}:\t", gt_pairs.len(), qid );
    gt_pairs
        .get(qid)
        .unwrap()
        .iter()
        .take(5)
        .for_each(|pair| {
            print!("{} d: {} ", pair.index, pair.distance );
        } );
    println!();

}

fn do_queries(
    queries: Vec<Bsp<2>>,
    descent: &RDescentMatrix,
    dao: Rc<Dao<Bsp<2>>>,
    gt_pairs: &Vec<Vec<Pair>>,
    nn_table: &Array2<usize>,
    distance: fn(&Bsp<2>, &Bsp<2>) -> f32,
) {
    queries.
        iter().
        enumerate()
        .for_each( | (qid,query) | {
            let now = Instant::now();
            let (dists,qresults) =  descent.knn_search( query.clone(), dao.clone(), 100, distance );
            let (dists, qresults ) = ADD_ONE_TO_RESULTS(dists, qresults);
            let after = Instant::now();
            println!("Results for Q{}....", qid);
            println!("Time per query: {} ns", (after - now).as_nanos());
            println!("Number of results = {} ", qresults.len() );
            println!("Dists: {:?}", dists);
            show_results(qid,&qresults);
            show_gt(qid,gt_pairs);
            println!("Number of GT results = {} ", gt_pairs[0].len() );
            println!( "Intersection size: {}", intersection_size(&qresults, gt_pairs.get(qid).unwrap())  );
            println!( "DCG: {}", ndcg(&qresults,
                                      &gt_pairs
                                          .get(qid)
                                          .unwrap()
                                          [0..30].into() ) );
        } );
}

fn intersection_size(results: &Vec<Pair>, gt_pairs: &Vec<Pair>) -> usize {
    let gt_indices : Vec<usize> = gt_pairs.iter().map(|pair| pair.index).collect();
    results.iter().filter_map( |pair| { if gt_indices.contains(&pair.index) { Some(1) } else { None } } ) .count()
}

fn ADD_ONE_TO_RESULTS(length: usize, results: Vec<Pair>) -> (usize, Vec<Pair>) {

    let adjusted_results = results.into_iter().map( |pair| { Pair::new( pair.distance, pair.index + 1 ) } ).collect();

    (length, adjusted_results)

}



