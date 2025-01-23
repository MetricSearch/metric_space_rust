//! This implementation of the Descent algorithm in Rust
//! Transcribed as a learning exercise from PynnDescent.

mod heap;

use ::dao::Dao;
use crate::descent::heap::Heap;
use std::cmp::Ordering;
//use ndarray::{s, Array, ArrayBase, ArrayView, ArrayView1, Axis, Dim, OwnedRepr, SliceInfo};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::cmp::min;
use std::fmt::{Debug, Display, Formatter};
//use std::env::current_dir;
use std::iter;
use std::rc::Rc;
use itertools::Itertools;
use ndarray::Array;
use metrics::euc;
use rp_forest::tree::RPForest;
use utils::{arg_sort, arg_sort_2D};

pub struct Descent {
    pub current_graph: Heap,
}

impl Descent {
    pub fn new(dao: Rc<Dao>, num_neighbours: usize, use_rp_tree: bool) -> Descent {
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
}

/// This code actually performs the descent refining the Heap data structures to create better and better NN tables
fn nn_descent(
    current_graph: &mut Heap,
    dao: Rc<Dao>,
    rng: &mut ChaCha8Rng,
    num_neighbours: usize,
    max_candidates: usize,
    num_iters: usize,
) {
    let num_vertices = dao.num_data;
    let block_size = 16384;
    let num_blocks = num_vertices / block_size;
    let delta = 0.001;

    let mut nn_table = dedup(&current_graph.indices); // these are the deduped neighbour indices - this is acopy

    for n in 0..num_iters {
        // outer loop which performs NN improvement
        println!("\t {} / {}", n + 1, num_iters);

        let (new_candidate_neighbors, old_candidate_neighbors) =
            new_build_candidates(current_graph, max_candidates, num_neighbours, num_vertices, rng); // a pair of neighbour graphs of the new and old

        let mut count_updates = 0;

        for i in 0..num_blocks {
            let block_start = i * block_size;
            let block_end = min(num_vertices, (i + 1) * block_size);

            let new_candidate_block =  &new_candidate_neighbors[block_start..block_end]; // horizontal slice out the block all columns
            let old_candidate_block =  &old_candidate_neighbors[block_start..block_end];

            let updates = generate_graph_updates(
                new_candidate_block,
                old_candidate_block,
                current_graph,
                dao.clone(),
            );

            count_updates = count_updates + apply_graph_updates(current_graph, updates, &mut nn_table);

            if count_updates as f32 <= delta * num_neighbours as f32 * num_vertices as f32 {
                println!(
                    "\tStopping threshold met -- exiting after, {}, iterations",
                    n + 1
                );
                return;
            }
        }
    }
}

fn init_rp_forest(dao: Rc<Dao>, num_neighbours: usize) -> Heap {
    println!("init_rp_forest");
    let forest = RPForest::new(30, 40, dao.clone() );
    let mut current_graph = Heap::new(dao.num_data, num_neighbours);

    for row in 0..dao.num_data {
        if row % 1_000 == 0 {
            print!(".");
        }
        if row % 10_000 == 0 {
            println!( "\nInitialised {} rows", row);
        }
        let neighbour_indices = forest.lookup(&dao.get(row));

        let neighbour_dists = neighbour_indices
            .iter()
            .map( |x| euc(dao.get(row).view(), dao.get(*x).view() ) )
            .collect::<Vec<f32>>();

        let (nns_indirect,nn_dists) = arg_sort(neighbour_dists.clone());

        (0 ..= num_neighbours) // Don't do index 0 (which is itself)   Has to be = iter because it may get itself
            .for_each( |nth_closest_indirect| {
                let neighbour_index = nns_indirect[nth_closest_indirect];
                let dao_index = neighbour_indices[neighbour_index];

                if dao_index != row {
                    checked_flagged_heap_push(&mut current_graph.indices[row], &mut current_graph.distances[row], &mut current_graph.flags[row], &nn_dists[nth_closest_indirect], dao_index as i32, 1);
                }
            } );

    }

    current_graph
}

fn init_random(dao: Rc<Dao>, num_neighbours: usize, rng: &mut ChaCha8Rng) -> Heap {
    let mut current_graph = Heap::new(dao.num_data, num_neighbours);
    let num_data = dao.num_data;

    for row in 0..num_data {
        for _ in 0..num_neighbours {
            // Stops duplicate entries in row or row containing itself in nn table
            let mut index = rng.gen_range(0..num_data);

            while index == row || current_graph.indices[row].iter().contains(&(index as i32)) {
                index = rng.gen_range(0..num_data);
            }

            let dist = euc(dao.get(index).view(), dao.get(row).view());
            let flag = 1;

            checked_flagged_heap_push(&mut current_graph.indices[row], &mut current_graph.distances[row], &mut current_graph.flags[row], &dist, index as i32, flag);
        }
    }

    current_graph
}

fn apply_graph_updates(current_heap: &mut Heap, updates: Vec<Vec<Update>>, nn_table: &mut Vec<Vec<i32>>) -> usize {
    let mut num_changes = 0;

    for i in 0..updates.len() {
        for j in 0..updates[i].len() {
            let Update(p, q, dist) = updates[i][j];  // TODO change to x,y
            if p == -1 || q == -1 || p == q { // not set or both equal
                continue;
            }

            if dist == f32::MAX { // should never happen
                panic!("Found a MAX dist when applying graph updates")
            }

            if nn_table[p as usize].contains(&q) && nn_table[q as usize].contains(&p) { // neighbours of q contains p and neighbours of p contains q
                continue;
            }
            if nn_table[p as usize].contains(&q) { // neighbours of q contains p
            } else {
                let added = checked_flagged_heap_push(&mut current_heap.indices[p as usize], &mut current_heap.distances[p as usize], &mut current_heap.flags[p as usize], &dist, q, 1);

                if added > 0 {
                    nn_table[p as usize].push(q);
                }
                num_changes += added
            }

            if p == q || nn_table[q as usize].contains(&p) {
            } else {
                let added = checked_flagged_heap_push(&mut current_heap.indices[p as usize], &mut current_heap.distances[p as usize], &mut current_heap.flags[p as usize], &dist, q, 1, );

                if added > 0 {
                    nn_table[q as usize].push(p);
                }
                num_changes += added
            }
        }
    }

    num_changes
}

fn generate_graph_updates(
    new_candidate_block: &[Vec<i32>],
    old_candidate_block: &[Vec<i32>],
    current_graph: &mut Heap,
    dao: Rc<Dao>
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

                let ac_dist = euc(dao.get(a as usize).view(), dao.get(c as usize).view());
                if ac_dist <= distances[a as usize][0] || ac_dist <= distances[c as usize][0] {      // first entry in the distances is the highest?
                    updates[b].push(Update(a as i32, c as i32, ac_dist));
                }
            }

            for column_index2 in 0..max_candidates {
                let c = old_candidate_block[b][column_index2];
                if c < 0 {
                    continue;
                }
                let dist = euc(dao.get(a as usize).view(), dao.get(c as usize).view());
                if dist <= distances[a as usize][0] || dist <= distances[c as usize][0] {   // first entry in the distances is the highest?
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
            row
            .into_iter()
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
    rng: &mut ChaCha8Rng
) -> (
    Vec<Vec<i32>>,
    Vec<Vec<i32>>,
) {

    let current_indices = &current_graph.indices;
    let current_flags = &current_graph.flags;

    let mut new_candidate_indices : Vec<Vec<i32>> = vec![vec![-1;max_candidates]; num_vertices]; // build a new array n_vertices X max_candidates of indices of -1 = not connected
    let mut new_candidate_distances: Vec<Vec<f32>> = vec![vec![f32::MAX; max_candidates]; num_vertices]; // build a new array n_vertices X max_candidates of infinity
    let mut old_candidate_indices : Vec<Vec<i32>> = vec![vec![-1;max_candidates]; num_vertices]; // build a new array n_vertices X max_candidates of indices of -1 = not connected
    let mut old_candidate_distances: Vec<Vec<f32>> = vec![vec![f32::MAX; max_candidates]; num_vertices]; // build a new array n_vertices X max_candidates of infinity

    // for n in numba.prange(n_threads): TODO fix concurrency

    for row_index in 0..num_vertices {          // iterate through the current indices
        for column_index in 0..num_neighbors {
            //let friend_index =
            let friend_index = current_indices[row_index][column_index]; // a friend from row_index,column_index (a dao index)
            if friend_index < 0 { // -1 represents data nor present
                continue
            }

            let friend_index = friend_index as usize; // we have now checked it is not -1 can make it usize

            let priority = rng.gen_range(0.0..f32::MAX); // a random number - used to sort the data when pushed

            let is_new = current_flags[row_index][column_index];

            if is_new != 0 { // if the point at position j for row i is new so add to new_candidates
                // TODO  thread code here in Python version
                // puts row_index into the candidates for the friend and friend_index into the candidates for the row
                // this is where the bi-directionality comes from
                checked_heap_push(&mut new_candidate_distances[row_index], &mut new_candidate_indices[row_index], &priority, &friend_index);    // push the friend_index into the candidates for row
                checked_heap_push(&mut new_candidate_distances[friend_index], &mut new_candidate_indices[friend_index], &priority, &row_index); // push the row index into the candidates for the friend
            } else { // the point at position j for row i is already in the set - add the row to the old friend only.
                // TODO  thread code here in Python version
                // remember the old state of the world here
                checked_heap_push(&mut old_candidate_distances[friend_index], &mut old_candidate_indices[friend_index], &priority, &row_index); // push the row_index into the candidates or the friend
            }

        }
    }

    let indices = &current_graph.indices;
    let flags = &mut current_graph.flags;

    // next clear the flags for all entries that are already present in the Heap

    for row_index in 0..num_vertices {  // iterate through the current indices
        for column_index in 0..num_neighbors {
            let friend_dao_index = indices[row_index][column_index];    // index - neighbour id, -1 if not a neighbour

            for cand_index in 0..max_candidates {
                let cand_dao_index = new_candidate_indices[row_index][cand_index];
                if cand_dao_index == friend_dao_index {                      // if already in the index
                    flags[row_index][column_index] = 0;                      // clear the new flag
                }
            }
        }
    }

    (new_candidate_indices, old_candidate_indices)
}

fn checked_heap_push(priorities: &mut Vec<f32>, indices: &mut Vec<i32>, priority: &f32, dao_index: &usize) -> bool {
    if priority >= &priorities[0] {
        false
    } else {
        priorities[0] = *priority; // insert the new priority in place of the furthest
        priorities.sort_by(|a,b| { b.partial_cmp(a).unwrap() }); // get the new entry into the right position
        let insert_position = priorities.iter().position(|&x| x == *priority).unwrap(); // find out where it went

        indices.insert(insert_position+1, *dao_index as i32); // insert into the rest of the indices - ignore the zeroth
        indices.remove(0); // remove the old first index

        true
    }
}

// Tom was here

fn checked_flagged_heap_push(indices: &mut Vec<i32>, priorities: &mut Vec<f32>, flags: &mut Vec<u8>, dist: &f32, index: i32, flag: u8) -> usize {
    if dist >= &priorities[0] {
        return 0 // dist greater than furthest distance return no updates
    }

    // break if we already have this element.
    for i in 0..priorities.len() {
        if index == indices[i] {
            return 0 // already got this entry - no update
        }
    }

    priorities[0] = *dist; // insert the new priority in place of the furthest
    priorities.sort_by(|a,b| { b.partial_cmp(a).unwrap() }); // get the new entry into the right position // TODO look at this too
    let insert_position = priorities.iter().position(|&x| x == *dist).unwrap(); // find out where it went

    indices.insert(insert_position+1, index); // insert into the rest of the indices - ignore the zeroth
    indices.remove(0); // remove the old first index in vector

    flags.insert(insert_position+1, flag);
    flags.remove(0);  // remove the old first flag in vector

    1 // one update
}


impl Debug for Descent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "indices:\n{}\n\ndistances:{}",
               to_string_indices(&self.current_graph.indices),
               to_string_distances(&self.current_graph.distances) )
    }
}

fn to_string_indices(indices: &Vec<Vec<i32>>) -> String {
    indices
        .iter()
        .map(|row| format!("[{}]", {
            row.iter()
                .map(|&x| x.to_string())
                .join(", ")} ))
        .join("\n")
}

fn to_string_distances(indices: &Vec<Vec<f32>>) -> String {
    indices
        .iter()
        .map(|row| format!("[{}]", {
            row.iter()
                .map(|nn| nn)
                .join(", ")} ))
        .join("\n")
}

struct Update(i32, i32, f32);





