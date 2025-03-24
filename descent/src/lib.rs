//! This implementation of the Descent algorithm in Rust
//! Transcribed as a learning exercise from PynnDescent.

mod heap;
pub mod non_nan;
pub mod pair;

use dao::Dao;

use crate::heap::Heap;
use crate::non_nan::NonNan;
use crate::pair::Pair;
use dao::DataType;
use itertools::Itertools;
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use rp_forest::tree::RPForest;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::cmp::{min, Ordering};
use std::collections::{BTreeSet, BinaryHeap};
use std::fmt::Debug;
use std::iter;
use std::rc::Rc;
use utils::arg_sort;

#[derive(Serialize, Deserialize)]
pub struct Descent {
    pub current_graph: Heap,
}

impl Descent {
    pub fn new<T: Clone + DataType>(
        dao: Rc<Dao<T>>,
        num_neighbours: usize,
        use_rp_tree: bool,
    ) -> Descent {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(324 * 142); // random number
        let mut current_graph = if use_rp_tree {
            init_rp_forest(dao.clone(), num_neighbours)
        } else {
            init_random(dao.clone(), num_neighbours, &mut rng)
        };
        let max_candidates = 50;
        let num_iters = 10;
        nn_descent(
            &mut current_graph,
            dao,
            &mut rng,
            num_neighbours,
            max_candidates,
            num_iters,
        );
        Self { current_graph }
    }

    /*
      Take the heap and transform it using RNG* algorithm
      Heap is already in NN order
    */
    pub fn rng_star<T: Clone + DataType>(&self, dao: Rc<Dao<T>>) -> Vec<Vec<usize>> {
        let num_entries = self.current_graph.num_entries;
        println!("num entries: {}", num_entries);

        let mut result: Vec<Vec<usize>> = vec![];

        (0..num_entries).for_each(|row_index| {
            let mut rng_star: Vec<usize> = vec![];
            let nns = &self.current_graph.nns[row_index];
            let first = nns[0]; // get the closest item in the row
            rng_star.push(first as usize); // push on the nearest neighbour

            for neighbour_index in 1..nns.len() {
                let next_neighbour_id = nns[neighbour_index];
                let next_neighbour = dao.get_datum(next_neighbour_id as usize);
                let q_to_next_neighbour_dist =
                    self.current_graph.distances[row_index][neighbour_index]; //T::dist(q, next_datum);

                let mut found_closer = false;
                // considering adding next_data to rng_star
                // will add it       if it is closer to q than any member of RNG star
                // i.e. will NOT add if it is closer any member of RNG star than to q.
                for id_already_in_rng in &rng_star {
                    let datum_already_in_rng = dao.get_datum(*id_already_in_rng);
                    let rng_datum_to_next_neighbour_dist =
                        T::dist(datum_already_in_rng, next_neighbour);
                    if rng_datum_to_next_neighbour_dist < q_to_next_neighbour_dist {
                        // this is the don't add condition
                        found_closer = true;
                        break;
                    }
                }
                if !found_closer {
                    rng_star.push(next_neighbour_id as usize);
                }
            } // end for
            result.push(rng_star);
        });

        result
    }

    pub fn knn_search<T: Clone + DataType>(
        &self,
        query: T,
        nn_table: &Vec<Vec<usize>>,
        dao: Rc<Dao<T>>,
        swarm_size: usize,
    ) -> (usize, Vec<(Pair)>) {
        let entry_point_simple = get_entry_point(&nn_table);
        //println!("getting entry point");
        let entry_point_good = find_good_entry_point(&query, dao.clone(), 100);
        println!("doing search with: {entry_point_simple} {entry_point_good}");
        return knn_search_internal(query, nn_table, dao, entry_point_simple, swarm_size);
    }
}

fn find_good_entry_point<T: Clone + DataType>(
    query: &T,
    dao: Rc<Dao<T>>,
    how_many: usize,
) -> usize {
    dao.get_data().iter().enumerate().take(how_many).fold(
        (0, f32::MAX),
        |(min_index, min_dist), (index, data_point)| {
            let d = T::dist(data_point, &query);
            if d < min_dist {
                (index, d)
            } else {
                (min_index, min_dist)
            }
        }
    ).0
}
/******* Private below here *******/

fn get_entry_point(nn_table: &Vec<Vec<usize>>) -> usize {
    return 100; // nn_table.len() / 4;
}

/**
 * query - the query to perform
 * nn_table - the nn table created by running rng_star on Descent
 * entry_point_index - a starting index into the NN table
 * @param ef - the amount of backtracking required - size of results list
 * @return nn nodes closest to query, this list is ordered with the closest first.
 */
fn knn_search_internal<T: Clone + DataType>(
    query: T,
    nn_table: &Vec<Vec<usize>>,
    dao: Rc<Dao<T>>,
    entry_point: usize,
    ef: usize,
) -> (usize, Vec<Pair>) {
    let mut visited_set: BTreeSet<usize> = BTreeSet::new();

    let ep_q_dist = NonNan(T::dist(&query, dao.get_datum(entry_point)));

    let mut results_list: BinaryHeap<Pair> = BinaryHeap::new(); // biggest first - a max-heap
    let mut candidates_list: BinaryHeap<Reverse<Pair>> = BinaryHeap::new(); // in reverse order - smallest first
    candidates_list.push(Reverse(Pair::new(ep_q_dist, entry_point)));
    visited_set.insert(entry_point);

    let mut finished = false;

    while !finished {
        if candidates_list.len() == 0 {
            finished = true;
        } else {
            let nearest_candidate_pair = candidates_list.pop().unwrap().0; // 0 to take Reverse  - get the nearest element to q (sort order for C) - extract the reverse

            if !results_list.is_empty()
                && nearest_candidate_pair.distance > results_list.peek().unwrap().distance
            {
                // if furthest than the furthest we are done
                finished = true;
            } else {
                results_list.push(Pair::new(
                    nearest_candidate_pair.distance,
                    nearest_candidate_pair.index,
                )); // should be a copy perhaps?
                if results_list.len() > ef {
                    // might not be full so check length after push
                    results_list.pop();
                }
                let neighbours_of_nearest_candidate = &nn_table[nearest_candidate_pair.index]; // List<Integer> - nns of nearest_candidate

                let new_cands: Vec<Reverse<Pair>> = neighbours_of_nearest_candidate
                    .into_iter()
                    .map(|&x| (x, dao.get_datum(x)))
                    .into_iter()
                    .filter_map(|neighbour_index| {
                        if visited_set.contains(&neighbour_index.0) {
                            None
                        } else {
                            visited_set.insert(neighbour_index.0);
                            Some(neighbour_index)
                        }
                    })
                    .map(|unseen_neighbour| {
                        let distance_q_next_neighbour =
                            NonNan(T::dist(&query, &unseen_neighbour.1));
                        Reverse(Pair::new(distance_q_next_neighbour, unseen_neighbour.0))
                    })
                    .collect();

                candidates_list.extend(new_cands);
            }
        }

        // println!("visited: {} candidates: {}", visited_set.len(), candidates_list.len());
    }

    return (candidates_list.len(), results_list.into_sorted_vec()); /* distances plus Vec<Pair> */
}

/// This code actually performs the descent refining the Heap data structures to create better and better NN tables
fn nn_descent<T: Clone + DataType>(
    current_graph: &mut Heap,
    dao: Rc<Dao<T>>,
    rng: &mut ChaCha8Rng,
    num_neighbours: usize,
    max_candidates: usize,
    num_iters: usize,
) {
    let num_vertices = dao.num_data;
    let block_size = 16384;
    let num_blocks = num_vertices / block_size;
    let delta = 0.001;

    let mut nn_table = dedup(&current_graph.nns); // these are the deduped neighbour indices - this is acopy

    tracing::info!("Descent iterating...");
    for n in 0..num_iters {
        // outer loop which performs NN improvement
        tracing::info!("\t {} / {}", n + 1, num_iters);

        let (new_candidate_neighbors, old_candidate_neighbors) = new_build_candidates(
            current_graph,
            max_candidates,
            num_neighbours,
            num_vertices,
            rng,
        ); // a pair of neighbour graphs of the new and old

        let mut count_updates = 0;

        for i in 0..num_blocks {
            let block_start = i * block_size;
            let block_end = min(num_vertices, (i + 1) * block_size);

            let new_candidate_block = &new_candidate_neighbors[block_start..block_end]; // horizontal slice out the block all columns
            let old_candidate_block = &old_candidate_neighbors[block_start..block_end];

            let updates = generate_graph_updates(
                new_candidate_block,
                old_candidate_block,
                current_graph,
                dao.clone(),
            );

            count_updates =
                count_updates + apply_graph_updates(current_graph, updates, &mut nn_table);

            if count_updates as f32 <= delta * num_neighbours as f32 * num_vertices as f32 {
                tracing::info!(
                    "\tStopping threshold met -- exiting after, {}, iterations",
                    n + 1
                );
                break;
            }
        }
    }
    println!("Reordering Heap...");
    reorder(current_graph);
}

// This reorders the heap which is in furthest to nearest order into nearest to furthest order
fn reorder(current_graph: &mut Heap) {
    // first reverse all the rows:
    current_graph.nns.iter_mut().for_each(|row| row.reverse());
    current_graph.flags.iter_mut().for_each(|row| row.reverse());
    current_graph
        .distances
        .iter_mut()
        .for_each(|row| row.reverse());
    // next check for -1s in indices

    current_graph.nns.iter().for_each(|row| {
        if row[0] == -1 {
            panic!("-1 found in indices")
        }
    });
}

fn init_rp_forest<T: Clone + DataType>(dao: Rc<Dao<T>>, num_neighbours: usize) -> Heap {
    println!("init_rp_forest");
    let forest = RPForest::new(30, 40, dao.clone());
    let mut current_graph = Heap::new(dao.num_data, num_neighbours);

    for row in 0..dao.num_data {
        if row % 100_000 == 0 {
            tracing::info!("\nForest initialised {} rows", row);
        }
        let neighbour_indices = forest.lookup(dao.get_datum(row).clone());

        let neighbour_dists = neighbour_indices
            .iter()
            .map(|x| T::dist(dao.get_datum(row), dao.get_datum(*x)))
            .collect::<Vec<f32>>();

        let (nns_indirect, nn_dists) = arg_sort(neighbour_dists.clone());

        (0..=num_neighbours) // Don't do index 0 (which is itself)   Has to be = iter because it may get itself
            .for_each(|nth_closest_indirect| {
                let neighbour_index = nns_indirect[nth_closest_indirect];
                let dao_index = neighbour_indices[neighbour_index];

                if dao_index != row {
                    checked_flagged_heap_push(
                        &mut current_graph.nns[row],
                        &mut current_graph.distances[row],
                        &mut current_graph.flags[row],
                        &nn_dists[nth_closest_indirect],
                        dao_index as i32,
                        1,
                    );
                }
            });
    }

    current_graph
}

fn init_random<T: Clone + DataType>(
    dao: Rc<Dao<T>>,
    num_neighbours: usize,
    rng: &mut ChaCha8Rng,
) -> Heap {
    let mut current_graph = Heap::new(dao.num_data, num_neighbours);
    let num_data = dao.num_data;

    for row in 0..num_data {
        for _ in 0..num_neighbours {
            // Stops duplicate entries in row or row containing itself in nn table
            let mut index = rng.gen_range(0..num_data);

            while index == row || current_graph.nns[row].iter().contains(&(index as i32)) {
                index = rng.gen_range(0..num_data);
            }

            let dist = T::dist(dao.get_datum(index), dao.get_datum(row));
            let flag = 1;

            checked_flagged_heap_push(
                &mut current_graph.nns[row],
                &mut current_graph.distances[row],
                &mut current_graph.flags[row],
                &dist,
                index as i32,
                flag,
            );
        }
    }

    current_graph
}

fn apply_graph_updates(
    current_heap: &mut Heap,
    updates: Vec<Vec<Update>>,
    nn_table: &mut Vec<Vec<i32>>,
) -> usize {
    let mut num_changes = 0;

    for i in 0..updates.len() {
        for j in 0..updates[i].len() {
            let Update(p, q, dist) = updates[i][j]; // TODO change to x,y
            if p == -1 || q == -1 || p == q {
                // not set or both equal
                continue;
            }

            if dist == f32::MAX {
                // should never happen
                panic!("Found a MAX dist when applying graph updates")
            }

            if nn_table[p as usize].contains(&q) && nn_table[q as usize].contains(&p) {
                // neighbours of q contains p and neighbours of p contains q
                continue;
            }
            if nn_table[p as usize].contains(&q) { // neighbours of q contains p
            } else {
                let added = checked_flagged_heap_push(
                    &mut current_heap.nns[p as usize],
                    &mut current_heap.distances[p as usize],
                    &mut current_heap.flags[p as usize],
                    &dist,
                    q,
                    1,
                );

                if added > 0 {
                    nn_table[p as usize].push(q);
                }
                num_changes += added
            }

            if p == q || nn_table[q as usize].contains(&p) {
            } else {
                let added = checked_flagged_heap_push(
                    &mut current_heap.nns[p as usize],
                    &mut current_heap.distances[p as usize],
                    &mut current_heap.flags[p as usize],
                    &dist,
                    q,
                    1,
                );

                if added > 0 {
                    nn_table[q as usize].push(p);
                }
                num_changes += added
            }
        }
    }

    num_changes
}

fn generate_graph_updates<T: Clone + DataType>(
    new_candidate_block: &[Vec<i32>],
    old_candidate_block: &[Vec<i32>],
    current_graph: &mut Heap,
    dao: Rc<Dao<T>>,
) -> Vec<Vec<Update>> {
    let distances = &current_graph.distances;
    let block_size = new_candidate_block.len();
    let max_candidates = new_candidate_block[0].len();

    let mut updates: Vec<Vec<Update>> = vec![];
    // initialise the updates data structure with empty entries.
    for _ in 0..block_size {
        updates.push(vec![Update(-1, -1, f32::MAX)]);
    }

    // The names in this code assume the following:
    //         b --- a
    //         |    /
    //         |   /
    //         |  / ac_dist
    //         | /
    //         |/
    //         c

    for b in 0..block_size {
        for first_column_index in 0..max_candidates {
            let a = new_candidate_block[b][first_column_index];
            if a < 0 {
                continue;
            }

            for rest_column_index in first_column_index..max_candidates {
                let c = new_candidate_block[b][rest_column_index];
                if c < 0 {
                    continue;
                }

                let ac_dist = T::dist(dao.get_datum(a as usize), dao.get_datum(c as usize));
                if ac_dist <= distances[a as usize][0] || ac_dist <= distances[c as usize][0] {
                    // first entry in the distances is the highest?
                    updates[b].push(Update(a as i32, c as i32, ac_dist));
                }
            }

            for column_index2 in 0..max_candidates {
                let c = old_candidate_block[b][column_index2];
                if c < 0 {
                    continue;
                }
                let dist = T::dist(dao.get_datum(a as usize), dao.get_datum(c as usize));
                if dist <= distances[a as usize][0] || dist <= distances[c as usize][0] {
                    // first entry in the distances is the highest?
                    updates[b].push(Update(a as i32, c as i32, dist));
                }
            }
        }
    }
    updates
}

fn dedup(indices: &Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    let indices = indices.clone(); // copy the indices
    let row_len = indices[0].len();

    indices
        .into_iter() // iter over the row contents
        .map(|row| {
            row.into_iter()
                .dedup()
                .chain(iter::repeat(-1))
                .take(row_len)
                .collect::<Vec<i32>>()
        })
        .collect()
}

/// Build a heap of candidate neighbors for nearest neighbor descent.
/// For each vertex the candidate neighbors are any current neighbors, and any
//  vertices that have the vertex as one of their nearest neighbors.
//  Returns a tuple containing the old and new candidate indices WHAT ARE THEY?
fn new_build_candidates(
    current_graph: &mut Heap,
    max_candidates: usize,
    num_neighbors: usize,
    num_vertices: usize,
    rng: &mut ChaCha8Rng,
) -> (Vec<Vec<i32>>, Vec<Vec<i32>>) {
    let current_indices = &current_graph.nns;
    let current_flags = &current_graph.flags;

    let mut new_candidate_indices: Vec<Vec<i32>> = vec![vec![-1; max_candidates]; num_vertices]; // build a new array n_vertices X max_candidates of indices of -1 = not connected
    let mut new_candidate_distances: Vec<Vec<f32>> =
        vec![vec![f32::MAX; max_candidates]; num_vertices]; // build a new array n_vertices X max_candidates of infinity
    let mut old_candidate_indices: Vec<Vec<i32>> = vec![vec![-1; max_candidates]; num_vertices]; // build a new array n_vertices X max_candidates of indices of -1 = not connected
    let mut old_candidate_distances: Vec<Vec<f32>> =
        vec![vec![f32::MAX; max_candidates]; num_vertices]; // build a new array n_vertices X max_candidates of infinity

    // for n in numba.prange(n_threads): TODO fix concurrency

    for row_index in 0..num_vertices {
        // iterate through the current indices
        for column_index in 0..num_neighbors {
            //let friend_index =
            let friend_index = current_indices[row_index][column_index]; // a friend from row_index,column_index (a dao index)
            if friend_index < 0 {
                // -1 represents data nor present
                continue;
            }

            let friend_index = friend_index as usize; // we have now checked it is not -1 can make it usize

            let priority = rng.gen_range(0.0..f32::MAX); // a random number - used to sort the data when pushed

            let is_new = current_flags[row_index][column_index];

            if is_new != 0 {
                // if the point at position j for row i is new so add to new_candidates
                // TODO  thread code here in Python version
                // puts row_index into the candidates for the friend and friend_index into the candidates for the row
                // this is where the bi-directionality comes from
                checked_heap_push(
                    &mut new_candidate_distances[row_index],
                    &mut new_candidate_indices[row_index],
                    &priority,
                    &friend_index,
                ); // push the friend_index into the candidates for row
                checked_heap_push(
                    &mut new_candidate_distances[friend_index],
                    &mut new_candidate_indices[friend_index],
                    &priority,
                    &row_index,
                ); // push the row index into the candidates for the friend
            } else {
                // the point at position j for row i is already in the set - add the row to the old friend only.
                // TODO  thread code here in Python version
                // remember the old state of the world here
                checked_heap_push(
                    &mut old_candidate_distances[friend_index],
                    &mut old_candidate_indices[friend_index],
                    &priority,
                    &row_index,
                ); // push the row_index into the candidates or the friend
            }
        }
    }

    let indices = &current_graph.nns;
    let flags = &mut current_graph.flags;

    // next clear the flags for all entries that are already present in the Heap

    for row_index in 0..num_vertices {
        // iterate through the current indices
        for column_index in 0..num_neighbors {
            let friend_dao_index = indices[row_index][column_index]; // index - neighbour id, -1 if not a neighbour

            for cand_index in 0..max_candidates {
                let cand_dao_index = new_candidate_indices[row_index][cand_index];
                if cand_dao_index == friend_dao_index {
                    // if already in the index
                    flags[row_index][column_index] = 0; // clear the new flag
                }
            }
        }
    }

    (new_candidate_indices, old_candidate_indices)
}

fn checked_heap_push(
    priorities: &mut Vec<f32>,
    indices: &mut Vec<i32>,
    priority: &f32,
    dao_index: &usize,
) -> bool {
    if priority >= &priorities[0] {
        false
    } else {
        priorities[0] = *priority; // insert the new priority in place of the furthest
        priorities.sort_by(|a, b| b.partial_cmp(a).unwrap()); // get the new entry into the right position
        let insert_position = priorities.iter().position(|&x| x == *priority).unwrap(); // find out where it went

        indices.insert(insert_position + 1, *dao_index as i32); // insert into the rest of the indices - ignore the zeroth
        indices.remove(0); // remove the old first index

        true
    }
}

fn checked_flagged_heap_push(
    indices: &mut Vec<i32>,
    priorities: &mut Vec<f32>,
    flags: &mut Vec<u8>,
    dist: &f32,
    index: i32,
    flag: u8,
) -> usize {
    if dist >= &priorities[0] {
        return 0; // dist greater than furthest distance return no updates
    }

    // break if we already have this element.
    for i in 0..priorities.len() {
        if index == indices[i] {
            return 0; // already got this entry - no update
        }
    }

    priorities[0] = *dist; // insert the new priority in place of the furthest
    priorities.sort_by(|a, b| b.partial_cmp(a).unwrap()); // get the new entry into the right position // TODO look at this too
    let insert_position = priorities.iter().position(|&x| x == *dist).unwrap(); // find out where it went

    indices.insert(insert_position + 1, index); // insert into the rest of the indices - ignore the zeroth
    indices.remove(0); // remove the old first index in vector

    flags.insert(insert_position + 1, flag);
    flags.remove(0); // remove the old first flag in vector

    1 // one update
}

impl Debug for Descent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "indices:\n{}\n\ndistances:{}",
            to_string_indices(&self.current_graph.nns),
            to_string_distances(&self.current_graph.distances)
        )
    }
}

fn to_string_indices(indices: &Vec<Vec<i32>>) -> String {
    indices
        .iter()
        .map(|row| format!("[{}]", { row.iter().map(|&x| x.to_string()).join(", ") }))
        .join("\n")
}

fn to_string_distances(indices: &Vec<Vec<f32>>) -> String {
    indices
        .iter()
        .map(|row| format!("[{}]", { row.iter().map(|nn| nn).join(", ") }))
        .join("\n")
}

struct Update(i32, i32, f32);
