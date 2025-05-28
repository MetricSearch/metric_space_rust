use anyhow::Result;
use dao::convert_f32_to_cubic::to_cubic_dao;
use dao::csv_dao_loader::dao_from_csv_dir;
use dao::Dao;
use ndarray::Array1;
use std::rc::Rc;
use utils::index::Index;
use wide::u64x4;
//use tracing_subscriber::EnvFilter;
use bits::whamming_distance;
use bitvec_simd::BitVecSimd;
use metrics::euc;
use std::time::Instant;

fn main() -> Result<()> {
    println!("Hello from Hamming Brute Force");
    tracing_subscriber::fmt::init();
    // let filter = EnvFilter::from_default_env()
    //     .add_directive("debug".parse().unwrap())
    //     .add_directive("rp_forest=warn".parse().unwrap());
    // tracing_subscriber::fmt().with_env_filter(filter).init();
    //let now = Instant::now();
    tracing::info!("Loading mf dino data...");
    let num_queries = 10_000; // for runnning: 10_000;  // for testing 990_000
    let num_data = 1_000_000 - num_queries;
    let dao_f32: Rc<Dao<Array1<f32>>> = Rc::new(dao_from_csv_dir(
        "/Volumes/Data/RUST_META/mf_dino2_csv/",
        num_data,
        num_queries,
    )?);
    let num_neighbours = 100;
    //let max_candidates = 50;

    let dao_hamming: Rc<Dao<BitVecSimd<[u64x4; 4], 4>>> = to_cubic_dao(dao_f32.clone());

    let mut indices_f32 = vec![Index::MAX; num_neighbours]; // build a new array of nns
    let mut distances_f32 = vec![f32::MAX; num_neighbours]; // build a new array of infinity distances

    for query_index in 0..10 {
        let query_index = Index::from(query_index);
        let query_f32 = dao_f32.get_query(query_index);
        let query_bits = dao_hamming.get_query(query_index);

        for i in 0..num_data {
            let i = Index::from(i);
            let data = dao_f32.get_datum(i);
            let dist = euc(&query_f32, &data);
            add_to_nns(&mut distances_f32, &mut indices_f32, dist, i);
        }

        distances_f32.reverse();
        indices_f32.reverse();
        // let first_10_indices = &indices_f32[0..10];
        // let first_10_distances = &distances_f32[0..10];
        // println!("Euc NNs of Q({}):", query_index);
        // for i in 0..10 {
        //     println!("{} euc dist {}", first_10_indices[i], first_10_distances[i]);
        // }

        let mut indices_bits = vec![Index::MAX; num_neighbours]; // build a new array of nns
        let mut distances_bits = vec![f32::MAX; num_neighbours]; // build a new array of infinity distances

        let now = Instant::now();

        for i in 0..num_data {
            let i = Index::from(i);
            let data = dao_hamming.get_datum(i);
            let dist = whamming_distance(&query_bits, &data);
            add_to_nns(&mut distances_bits, &mut indices_bits, (dist as f32), i);
        }

        let elapsed = now.elapsed();
        println!("Hamming Elapsed: {:.2?}", elapsed);
        println!(
            "Distances per second: {}",
            num_data as f64 / elapsed.as_secs_f64()
        );

        let num_nns = 20;

        distances_bits.reverse();
        indices_bits.reverse();
        let first_indices = &indices_bits[0..num_nns];
        let first_distances = &distances_bits[0..num_nns];
        println!("Hamming NNs of Q({}):", query_index);
        for i in 0..num_nns {
            println!(
                "{} dist {} position {}",
                first_indices[i],
                first_distances[i],
                get_index(first_indices[i], &indices_f32)
            );
        }
    }

    Ok(())
}

fn get_index(p0: Index, true_nns: &[Index]) -> Index {
    true_nns
        .iter()
        .position(|&n| n == p0)
        .map(Index::from)
        .unwrap_or(Index::MAX)
}

fn add_to_nns(
    distances: &mut Vec<f32>,
    indices: &mut Vec<Index>,
    distance: f32,
    index: Index,
) -> bool {
    if distance >= distances[0] {
        false
    } else {
        distances[0] = distance; // insert the new priority in place of the furthest
        distances.sort_by(|a, b| b.partial_cmp(a).unwrap()); // get the new entry into the right position
        let insert_position = distances.iter().position(|&x| x == distance).unwrap(); // find out where it went

        indices.insert(insert_position + 1, index); // insert into the rest of the indices - ignore the zeroth
        indices.remove(0); // remove the old first index

        true
    }
}
