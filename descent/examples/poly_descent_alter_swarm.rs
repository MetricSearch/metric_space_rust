use anyhow::Result;
use bits::{
    f32_data_to_cubic_bitrep, f32_embedding_to_cubic_bitrep, hamming_distance,
    hamming_distance_as_f32, whamming_distance,
};
use bitvec_simd::BitVecSimd;
use metrics::euc;
use ndarray::{s, Array1, ArrayView1};
//use rayon::prelude::*;
use dao::convert_f32_to_cube_oct::to_cube_oct_dao;
use dao::convert_f32_to_cubic::to_cubic_dao;
use dao::csv_dao_loader::dao_from_csv_dir;
use dao::Dao;
use descent::Descent;
use std::fs::File;
use std::io::BufReader;
use std::rc::Rc;
use std::time::Instant;
use utils::non_nan::NonNan;
use utils::pair::Pair;
use utils::{arg_sort_2d, distance_f32, ndcg};
use wide::u64x4;
//use divan::Bencher;

fn main() -> Result<()> {
    tracing::info!("Loading mf dino data...");
    let num_queries = 10_000;
    let num_data = 1_000_000 - num_queries;

    let data_file_name = "/Volumes/Data/RUST_META/mf_dino2_csv/";
    let descent_file_name = "_scratch/nn_table_100.bin";
    let rng_star_file_name = "_scratch/rng_table_100.bin";

    println!("Poly search");
    println!("Serde load of Descent");
    let f = BufReader::new(File::open(descent_file_name).unwrap());
    let descent: Descent = bincode::deserialize_from(f).unwrap();

    println!("Loading mf dino data...");
    let num_queries = 10_000; // for runnning: 10_000;  // for testing 990_000
    let num_data = 1_000_000 - num_queries;

    let dao_f32: Rc<Dao<Array1<f32>>> =
        Rc::new(dao_from_csv_dir(data_file_name, num_data, num_queries)?);

    let dao_cube = to_cubic_dao(dao_f32.clone());
    let dao_cube_oct = to_cube_oct_dao(dao_f32.clone());

    let swarm_size = 100;
    let nn_table_size = 100;

    println!("f32:");
    run_with_dao(
        &descent,
        dao_f32.clone(),
        swarm_size,
        nn_table_size,
        distance_f32,
    );
    println!("cube:");
    run_with_dao(
        &descent,
        dao_cube.clone(),
        swarm_size,
        nn_table_size,
        hamming_distance_as_f32::<4>,
    );
    println!("cube oct:");
    run_with_dao(
        &descent,
        dao_cube_oct.clone(),
        swarm_size,
        nn_table_size,
        hamming_distance_as_f32::<4>,
    );

    Ok(())
}

fn run_with_dao<T: Clone>(
    descent: &Descent,
    dao: Rc<Dao<T>>,
    mut swarm_size: usize,
    nn_table_size: usize,
    distance: fn(&T, &T) -> f32,
) {
    while swarm_size > 0 {
        run_with_swarm(&descent, dao.clone(), swarm_size, nn_table_size, distance);
        swarm_size = swarm_size - 10;
    }
}

fn run_with_swarm<T: Clone>(
    descent: &Descent,
    dao: Rc<Dao<T>>,
    swarm_size: usize,
    mut nn_table_size: usize,
    distance: fn(&T, &T) -> f32,
) {
    let queries = dao.get_queries().to_vec();
    let data = dao.get_data().to_vec();

    let this_many_queries = 10;

    let (queries, _rest) = queries.split_at(this_many_queries);

    let gt_pairs: Vec<Vec<Pair>> = brute_force_all_dists(queries.to_vec(), data, distance);
    let nn_table = to_usize(&descent.current_graph.nns);

    println!("NNtable columns active {:?}", nn_table_size);
    println!("Swarm size {:?}", swarm_size);

    let nn_table = reduce_columns_to(nn_table, nn_table_size);

    println!("Doing {:?} queries", queries.len());

    println!("Running Queries");

    do_queries(
        queries,
        descent,
        dao.clone(),
        &gt_pairs,
        nn_table,
        swarm_size,
        distance,
    );
}

fn reduce_columns_to(nn_table: Vec<Vec<usize>>, num_columns: usize) -> Vec<Vec<usize>> {
    nn_table
        .iter()
        .map(|row| row.iter().cloned().take(num_columns).collect())
        .collect()
}

fn first_row(graph: &Descent) {
    let heap = &graph.current_graph;
    let row_indices = &heap.nns.get(0).unwrap();
    let dists = &heap.distances.get(0).unwrap();

    print!("First row of Descent table: ");
    row_indices
        .iter()
        .by_ref()
        .take(5)
        .zip(dists.iter())
        .for_each(|pair| {
            print!("{} d: {} ", pair.0, pair.1);
        })
}

/* checks that the distances are from low to high in the descent graph */
fn check_order(graph: &Descent) {
    let heap = &graph.current_graph;
    let dists = &heap.distances;
    let mut items_checked = 0;
    dists.iter().for_each(|row| {
        let mut val = row.get(0).unwrap();
        for i in row.iter() {
            if i < val {
                println!("Out of order {:?} !< {:?}", i, val);
                return;
            }
            val = i;
            items_checked = items_checked + 1;
        }
    });
    println!("distance order in graph all Ok: checked {}", items_checked);
}

fn show_results(qid: usize, results: &Vec<Pair>) {
    print!("first few results for q{}:\t", qid);
    results.iter().by_ref().take(5).for_each(|pair| {
        print!("{} d: {} ", pair.index, pair.distance.0);
    });
    println!();
}

fn show_gt(qid: usize, gt_pairs: &Vec<Vec<Pair>>) {
    //<<<<<<<<<<<<<<<<<
    print!("first few GT results for q{}:\t", qid);
    gt_pairs.get(qid).unwrap().iter().take(5).for_each(|pair| {
        print!("{} d: {} ", pair.index, pair.distance);
    });
    println!();
}

fn do_queries<T: Clone>(
    queries: &[T],
    descent: &Descent,
    dao: Rc<Dao<T>>,
    gt_pairs: &Vec<Vec<Pair>>,
    nn_table: Vec<Vec<usize>>,
    swarm_size: usize,
    distance: fn(&T, &T) -> f32,
) {
    queries.iter().enumerate().for_each(|(qid, query)| {
        let now = Instant::now();
        let (dists, qresults) =
            descent.knn_search(query.clone(), &nn_table, dao.clone(), swarm_size, distance);
        let after = Instant::now();
        print!("Q{} swarm_size, nn_table_size, time, dists, dcg\t", qid);
        print!("{}\t", qresults.len());
        print!("{}\t", nn_table.get(0).unwrap().len());
        print!("{}\t", (after - now).as_nanos());
        print!("{:?}\t", dists);
        // show_results(qid,&qresults);
        // show_gt(qid,gt_pairs);
        println!(
            "{}",
            ndcg(
                &qresults,
                &gt_pairs.get(qid).unwrap()[0..swarm_size - 1].into()
            )
        );
    });
}

// TODO fix this mess somehow!
fn to_usize(i32s: &Vec<Vec<i32>>) -> Vec<Vec<usize>> {
    i32s.into_iter()
        .map(|v| v.iter().map(|&v| v as usize).collect())
        .collect()
}

//Returns the nn(k)
fn brute_force_all_dists<T: Clone>(
    queries: Vec<T>,
    data: Vec<T>,
    distance: fn(&T, &T) -> f32,
) -> Vec<Vec<Pair>> {
    queries
        .iter()
        .map(|q| {
            let mut pairs = data
                .iter()
                .enumerate()
                .map(|it| Pair::new(NonNan(distance(q, it.1)), it.0))
                .collect::<Vec<Pair>>();
            pairs.sort(); // Pair has Ord _by( |a, b| { a.distance.0.cmp(  b.distance.0 ) } );
            pairs
        })
        .collect::<Vec<Vec<Pair>>>()
}
