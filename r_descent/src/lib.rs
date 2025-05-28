//! This implementation of Richard's NN table builder

mod updates;

use crate::updates::Updates;
use bits::{bsp_similarity_as_f32, matrix_dot_bsp, EvpBits};
use dao::{Dao, DaoMatrix};
use ndarray::parallel::prelude::*;
use ndarray::parallel::prelude::{IntoParallelIterator, IntoParallelRefIterator};
use ndarray::{
    concatenate, s, Array, Array1, Array2, ArrayBase, ArrayView1, ArrayView2, ArrayViewMut1, Axis,
    Ix1, Order, OwnedRepr,
};
use rand_chacha::rand_core::SeedableRng;
use rayon::prelude::*;
use rayon::spawn;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};
use std::ptr;
use std::rc::Rc;
use std::sync::atomic::AtomicUsize;
use std::time::Instant;
use utils::non_nan::NonNan;
use utils::pair::Pair;
use utils::{arg_sort_big_to_small_1d, bytes_fmt};
use utils::{arg_sort_big_to_small_2d, index_of_min, min_index_and_value, minimum_in, rand_perm};

const PARALLEL_WORK: usize = 1_000_000;

#[derive(Serialize, Deserialize)]
pub struct RDescentMatrix {
    pub neighbours: Array2<usize>,
    pub similarities: Array2<f32>,
}

pub trait IntoRDescent {
    fn into_rdescent(
        self: Rc<Self>,
        num_neighbours: usize,
        reverse_list_size: usize,
        chunk_size: usize,
        rho: f64,
        delta: f64,
    ) -> RDescentMatrix;
}

pub trait IntoRDescentWithRevNNs {
    fn into_rdescent_with_rev_nn(
        self: Rc<Self>,
        num_neighbours: usize,
        reverse_list_size: usize,
        chunk_size: usize,
        rho: f64,
        delta: f64,
        nns_in_search_structure: usize,
    ) -> RDescentMatrixWithRev;
}

pub struct RDescentMatrixWithRev {
    pub rdescent: RDescentMatrix,
    pub reverse_neighbours: Array2<usize>,
}

pub trait KnnSearch<T: Clone> {
    fn knn_search(
        &self,
        query: T,
        dao: Rc<Dao<T>>,
        num_neighbours: usize,
        distance: fn(&T, &T) -> f32,
    ) -> (usize, Vec<Pair>);
}

pub trait RevSearch<T: Clone> {
    fn rev_search(
        &self,
        query: T,
        dao: Rc<Dao<T>>,
        num_neighbours: usize,
        distance: fn(&T, &T) -> f32,
    ) -> (usize, Vec<Pair>);
}

//********** Impls **********

impl<T: Clone + Default + Hasher> KnnSearch<T> for RDescentMatrix {
    fn knn_search(
        &self,
        query: T,
        dao: Rc<Dao<T>>,
        num_neighbours: usize,
        distance: fn(&T, &T) -> f32,
    ) -> (usize, Vec<Pair>) {
        let mut visited_set: HashSet<usize, BuildHasherDefault<T>> = HashSet::default();
        let entry_point = 0; // <<<<<<<<<<<<<< TODO ENTRY POINT OF ZERO FOR NOW
        let ep_q_dist = NonNan::new(distance(&query, dao.get_datum(0)));
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
                    if results_list.len() > num_neighbours {
                        // was ef <<<<<<
                        // might not be full so check length after push
                        results_list.pop();
                    }
                    let neighbours_of_nearest_candidate =
                        &self.neighbours.row(nearest_candidate_pair.index); // List<Integer> - nns of nearest_candidate

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
                                NonNan::new(distance(&query, &unseen_neighbour.1));

                            // let distance_q_next_neighbour = dist_fn(&query, &unseen_neighbour.1);
                            Reverse(Pair::new(distance_q_next_neighbour, unseen_neighbour.0))
                        })
                        .collect();

                    candidates_list.extend(new_cands);
                }
            }
        }

        return (candidates_list.len(), results_list.into_sorted_vec()); /* distances plus Vec<Pair> */
    }
}

impl<const X: usize> RevSearch<EvpBits<X>> for RDescentMatrixWithRev {
    /* The function uses NN and revNN tables to query in the manner of descent
     * We start with a rough approximation of the query by selecting eg 1000 distances
     * Then we iterate to see if any of the NNs of these NNs are closer to the query, using the NN table directly but also the reverseNN table
     */

    fn rev_search(
        &self,
        query: EvpBits<X>,
        dao: Rc<Dao<EvpBits<X>>>,
        num_neighbours: usize,
        distance: fn(&EvpBits<X>, &EvpBits<X>) -> f32,
    ) -> (usize, Vec<Pair>) {
        let num_data = dao.num_data;
        let dims = dao.get_dim();
        let data = dao.get_data();

        // let mut result_indices = unsafe { Array2::<usize>::uninit((num_data, num_neighbours)).assume_init()};
        // let mut result_sims = unsafe {Array2::<f32>::uninit((num_data, num_neighbours)).assume_init()};

        // First, cheaply find some reasonably good solutions

        let query_as_array: ArrayBase<OwnedRepr<EvpBits<{ X }>>, Ix1> = Array1::from_elem(1, query);

        let data_subset = data.slice(s![0..1000]);
        let sims: Array2<f32> =
            matrix_dot_bsp::<X>(&data_subset, &query_as_array.view(), |a, b| {
                bsp_similarity_as_f32::<X>(a, b)
            }); // matrix mult all the distances - all relative to the original_rows
        let (ords, sims) = arg_sort_big_to_small_2d(&sims.view()); // ords are row relative indices - these are is 1 X 1000

        // // these ords are row relative all range from 0..1000 in data - so therefore real dao indices

        // // We need to initialise qNNs and qSims to start with, these will incrementally get better until the algoritm terminates

        let mut binding = ords.into_shape_with_order(1000).unwrap();
        let mut q_nns: ArrayViewMut1<usize> = binding.slice_mut(s![..num_neighbours]); // get these into a 1D array and take num_neighbours
        let mut binding = sims.into_shape_with_order(1000).unwrap();
        let mut q_sims: ArrayViewMut1<f32> = binding.slice_mut(s![..num_neighbours]); // get these into a 1D array and take num_neighbours

        let mut current_min_sim = -1.0;

        // same as in nnTableBuild, the new flags

        let mut new_flags: Array1<bool> = Array1::from_elem(q_nns.len(), true);

        // The amount of work done in the iteration

        let mut work_done = 1;

        while work_done > 0 {
            work_done = 0; // a bit strange?

            // q_nns are the current best NNs that we know about
            // but don't re-try ones that have already been added before this loop

            let selectors = get_selectors_from_flags(&new_flags);
            let these_q_nns: Array1<usize> = get_slice_using_selectors(
                &q_nns.view(),
                &selectors.view(),
                selectors.shape().try_into().unwrap(),
            );

            // set all to false; will be reset to true when overwritten with new values
            new_flags.fill(false);

            // get the friends of the new ones
            // forward_nns starts off as an X x k array if there are X these_q_nns

            // TODO what if these_q_nns is bigger than number of neighbours in the table

            let forward_nns: Array2<usize> =
                get_2D_slice_using(&self.rdescent.neighbours.view(), &these_q_nns.view());

            // and we make it into a one-dim Array
            let rows = forward_nns.nrows();
            let cols = forward_nns.ncols();
            let forward_nns = forward_nns.into_shape_with_order((rows * cols)).unwrap();

            // these two lines do the same for the reverse table as above for the forward table

            let reverse_nns: Array2<usize> =
                get_2D_slice_using(&self.reverse_neighbours.view(), &these_q_nns.view());
            let rows = reverse_nns.nrows();
            let cols = reverse_nns.ncols();
            let reverse_nns = reverse_nns.into_shape_with_order((rows * cols)).unwrap();

            // TODO eliminate duplicates from reverse_nns - but leave for now - expensive and tricky

            let all_ids = concatenate(Axis(0), &[forward_nns.view(), reverse_nns.view()]).unwrap();

            // TODO also zeros (which are not encoded as zeros)?

            // get a view of the actual data values from the full data set but not the zeros.

            let nn_data: Array1<EvpBits<X>> = get_1D_slice_using_selected(&data, &all_ids.view());

            // and measure the similarity of each to the query
            // allSims is a flat vector is distances
            // ie it is a 1 x N array where N is the number of elements in allIds
            // only it isnt so needs flattening
            let all_sims = matrix_dot_bsp(&nn_data.view(), &query_as_array.view(), |a, b| {
                bsp_similarity_as_f32::<X>(a, b)
            });

            let all_sims = all_sims.flatten();

            for i in 0..all_ids.len() {
                // this code is the same as just one of the four bits of phase 3 in the nn table build algorithm
                let this_id = all_ids[i];
                let this_sim = all_sims[i];
                // is the similarity of the query and thisId greater than the smallest similarity in the result set?
                // if it's not, then do nothing and carry on

                if this_sim > current_min_sim {
                    // more similar than what we have so far
                    if !q_nns.iter().any(|x| *x == this_id) {
                        // check if this_id is in the result set already
                        // and it's not, so we're doing a replacement
                        // first find where the current smallest similarity is
                        let (position, value) = min_index_and_value(&q_sims.view());
                        // then replace the id in the result list with the new id, also maintaining the global q_sims list
                        q_nns[position] = this_id;
                        q_sims[position] = this_sim;
                        new_flags[position] = true;
                        current_min_sim = current_min_sim.min(this_sim);
                        // and log that we've done some work so we don't want to stop yet
                        work_done += 1;
                    }
                }
            }
        }

        let (sorted_ords, sorted_dists) = arg_sort_big_to_small_1d(q_sims.view());
        let sorted_pairs = sorted_ords
            .iter()
            .map(|index| {
                Pair::new(
                    NonNan::new(sorted_dists[*index]),
                    q_nns[sorted_ords[*index]],
                )
            })
            .collect();

        (sorted_ords.len(), sorted_pairs) /* distances plus Vec<Pair> */
    }
}

fn get_selectors_from_flags(selectors: &Array1<bool>) -> Array1<usize> {
    let vec = selectors
        .into_iter()
        .enumerate()
        .filter_map(|(index, value)| if *value { Some(index) } else { None })
        .collect();

    Array1::from_vec(vec)
}

impl IntoRDescentWithRevNNs for DaoMatrix<f32> {
    fn into_rdescent_with_rev_nn(
        self: Rc<Self>,
        num_neighbours: usize,
        reverse_list_size: usize,
        chunk_size: usize,
        rho: f64,
        delta: f64,
        nns_in_search_structure: usize,
    ) -> RDescentMatrixWithRev {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(324 * 142);
        let (mut neighbours, mut similarities) =
            initialise_table_m(self.clone(), chunk_size, num_neighbours);
        get_nn_table2_m(
            self.clone(),
            &mut neighbours,
            &mut similarities,
            num_neighbours,
            rho,
            delta,
            reverse_list_size,
        );
        let (reverse_nns, reverse_similarities) =
            get_reverse_links_not_in_forward(&neighbours, &similarities, nns_in_search_structure);

        // let neighbours: Array2<usize> = concatenate(Axis(1), &[neighbours.view(), reverse_nns.view()]).unwrap();
        // let similarities: Array2<f32> = concatenate(Axis(1), &[similarities.view(), reverse_similarities.view()]).unwrap();

        // TODO perhaps need to deal with MAXINT values

        let r_descent = RDescentMatrix {
            neighbours: neighbours,
            similarities: similarities,
        };

        RDescentMatrixWithRev {
            rdescent: r_descent,
            reverse_neighbours: reverse_nns,
        }
    }
}

impl IntoRDescent for Dao<EvpBits<2>> {
    fn into_rdescent(
        self: Rc<Self>,
        num_neighbours: usize,
        reverse_list_size: usize,
        chunk_size: usize,
        rho: f64,
        delta: f64,
    ) -> RDescentMatrix {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(324 * 142);
        let (mut ords, mut dists) = initialise_table_bsp(self.clone(), chunk_size, num_neighbours);
        get_nn_table2_bsp(
            self.clone(),
            &mut ords,
            &mut dists,
            num_neighbours,
            rho,
            delta,
            reverse_list_size,
        );

        RDescentMatrix {
            neighbours: ords,
            similarities: dists,
        }
    }
}

impl IntoRDescentWithRevNNs for Dao<EvpBits<2>> {
    fn into_rdescent_with_rev_nn(
        self: Rc<Self>,
        num_neighbours: usize,
        reverse_list_size: usize,
        chunk_size: usize,
        rho: f64,
        delta: f64,
        nns_in_search_structure: usize,
    ) -> RDescentMatrixWithRev {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(324 * 142);
        let (mut neighbours, mut similarities) =
            initialise_table_bsp(self.clone(), chunk_size, num_neighbours);
        get_nn_table2_bsp(
            self.clone(),
            &mut neighbours,
            &mut similarities,
            num_neighbours,
            rho,
            delta,
            reverse_list_size,
        );
        let (reverse_nns, reverse_similarities) =
            get_reverse_links_not_in_forward(&neighbours, &similarities, nns_in_search_structure);

        let neighbours: Array2<usize> =
            concatenate(Axis(1), &[neighbours.view(), reverse_nns.view()]).unwrap();
        let similarities: Array2<f32> =
            concatenate(Axis(1), &[similarities.view(), reverse_similarities.view()]).unwrap();

        // TODO perhaps need to deal with MAXINT values

        RDescentMatrixWithRev {
            rdescent: RDescentMatrix {
                neighbours,
                similarities,
            },
            reverse_neighbours: todo!(), // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        }
    }
}

//********** Local impl fns **********

pub fn initialise_table_m(
    dao: Rc<DaoMatrix<f32>>,
    chunk_size: usize,
    num_neighbours: usize,
) -> (Array2<usize>, Array2<f32>) {
    let start_time = Instant::now();

    let num_data = dao.num_data;
    let dims = dao.get_dim();
    let data = dao.get_data();

    let mut result_indices =
        unsafe { Array2::<usize>::uninit((num_data, num_neighbours)).assume_init() };
    let mut result_sims =
        unsafe { Array2::<f32>::uninit((num_data, num_neighbours)).assume_init() };

    result_indices
        .axis_chunks_iter_mut(Axis(0), chunk_size)
        .into_par_iter() // .into_iter()
        .zip(result_sims.axis_chunks_iter_mut(Axis(0), chunk_size))
        .enumerate()
        .for_each(|(i, (mut result_indices_chunk, mut result_sims_chunk))| {
            let real_chunk_size = result_sims_chunk.shape()[0];
            let start_pos = i * chunk_size;
            let end_pos = start_pos + real_chunk_size;

            let chunk = data.slice(s![start_pos..end_pos, 0..]);

            let original_row_ids = rand_perm(num_data, real_chunk_size); // random data ids from whole data set
            let rand_data = get_slice_using_selected(
                &data,
                &original_row_ids.view(),
                chunk.shape().try_into().unwrap(),
            ); // a view of the original data points as a matrix
            let chunk_dists: Array2<f32> = chunk.dot(&rand_data.t()); // matrix mult all the distances - all relative to the original_rows

            let (sorted_ords, sorted_dists) = arg_sort_big_to_small_2d(&chunk_dists.view()); // sorted ords are row relative indices.
                                                                                             // these ords are row relative all range from 0..real_chunk_size

            // get the num_neighbours closest original data indices
            let mut closest_dao_indices: Array2<usize> =
                Array2::<usize>::zeros((real_chunk_size, num_neighbours));

            for row in 0..real_chunk_size {
                for col in 0..num_neighbours {
                    closest_dao_indices[[row, col]] = original_row_ids[sorted_ords[[row, col]]];
                }
            }

            result_indices_chunk.assign(&closest_dao_indices.slice(s![.., 0..num_neighbours]));
            result_sims_chunk.assign(&sorted_dists.slice(s![.., 0..num_neighbours]));
        });

    let end_time = Instant::now();
    log::debug!(
        "Initialistion in {:?}ms",
        ((end_time - start_time).as_millis() as f64)
    );

    (
        insert_index_at_position_1_inplace(result_indices),
        insert_column_inplace(result_sims, 1.0),
    )
}

pub fn get_nn_table2_m(
    dao: Rc<DaoMatrix<f32>>,
    mut neighbours: &mut Array2<usize>,
    mut similarities: &mut Array2<f32>, // bigger is better
    num_neighbours: usize,
    rho: f64,
    delta: f64,
    reverse_list_size: usize,
) {
    let start_time = Instant::now();

    let num_data = dao.num_data;
    let dims = dao.get_dim();
    let data = dao.get_data();

    // Matlab lines refer to richard_build.txt file in the matlab dir

    let mut iterations = 0;
    let mut neighbour_is_new = Array2::from_elem((num_data, num_neighbours), true);
    let mut work_done = num_data; // a count of the number of times a similarity minimum of row has changed - measure of flux

    while work_done > ((num_data as f64) * delta) as usize {
        // Matlab line 61
        // condition is fraction of lines whose min similarity has changed when this gets low - no much work done then stop.
        iterations += 1;

        log::debug!(
            "iterating: c: {} num_data: {} iters: {}",
            work_done,
            num_data,
            iterations
        );

        // phase 1

        let now = Instant::now();

        let mut new: Array2<usize> = Array2::from_elem((num_data, num_neighbours), 0); // Matlab line 65
        let mut old: Array2<usize> = Array2::from_elem((num_data, num_neighbours), 0);

        // initialise old and new inline

        for row in 0..num_data {
            // in Matlab line 74
            let row_flags = &neighbour_is_new.row_mut(row); // Matlab line 74

            // new_indices are the indices in this row whose flag is set to true (columns)

            let new_indices = row_flags // Matlab line 76
                .iter()
                .enumerate()
                .filter_map(|(index, flag)| if *flag { Some(index) } else { None })
                .collect::<Array1<usize>>();

            // old_indices are the indices in this row whose flag is set to false (intially there are none of these).

            let old_indices = row_flags // Matlab line 77
                .iter()
                .enumerate()
                .filter_map(|(index, flag)| if !*flag { Some(index) } else { None })
                .collect::<Array1<usize>>();

            // random data ids from whole data set
            // in matlab p = randperm(n,k) returns a row vector containing k unique integers selected randomly from 1 to n

            let sampled = rand_perm(
                new_indices.len(),
                (rho * (new_indices.len() as f64)).round() as u64 as usize,
            );

            // sampled are random indices from new_indices

            let mut new_row_view: ArrayViewMut1<usize> = new.row_mut(row);
            let mut neighbour_row_view: ArrayViewMut1<bool> = neighbour_is_new.row_mut(row);

            fill_selected(&mut new_row_view, &neighbours.row(row), &sampled.view()); // Matlab line 79
            fill_selected(&mut new_row_view, &neighbours.row(row), &old_indices.view());
            fill_false(&mut neighbour_row_view, &sampled.view())
        }

        let after = Instant::now();
        log::debug!("Phase 1: {} ms", ((after - now).as_millis() as f64));

        // phase 2  Matlab line 88

        let (reverse, _reverse_sims) = get_new_reverse_links_not_in_forward(
            &neighbours,
            &similarities,
            reverse_list_size,
            &new,
        );

        // phase 3

        let now = Instant::now();

        work_done = 0;

        let mut updates = Updates::new(num_data);

        old.axis_iter_mut(Axis(0)) // Get mutable rows (disjoint slices)
            .enumerate()
            .zip(new.axis_iter_mut(Axis(0)))
            .par_bridge()
            .map(|((row, old_row), new_row)| {
                let mut reverse_link_row: Array1<usize> = reverse
                    .row(row)
                    .iter()
                    .filter(|&&v| v != 0)
                    .copied()
                    .collect();

                if rho < 1.0 {
                    // Matlab line 127
                    // randomly shorten the reverse_link_row vector
                    let reverse_indices = rand_perm(
                        reverse_link_row.len(),
                        (rho * reverse_link_row.len() as f64).round() as usize,
                    );
                    reverse_link_row = reverse_indices
                        .iter()
                        .map(|&i| reverse_link_row[i])
                        .collect::<Array1<usize>>();
                }
                let mut new_row_union: Array1<usize> = if new_row.len() == 0 {
                    // Matlab line 130
                    Array1::from(vec![])
                } else {
                    new_row
                        .iter()
                        .copied()
                        .chain(reverse_link_row.iter().copied())
                        .collect::<Array1<usize>>() // <<<<< 2 row copies here
                };

                let new_row_union_len = new_row_union.len();

                // index the data using the rows indicated in old_row
                let old_data =
                    get_slice_using_selected(&data, &old_row.view(), [old_row.len(), dims]); // Matlab line 136
                let new_data =
                    get_slice_using_selected(&data, &new_row.view(), [new_row.len(), dims]); // Matlab line 137
                let new_union_data = get_slice_using_selected(
                    &data,
                    &new_row_union.view(),
                    [new_row_union_len, dims],
                ); // Matlab line 137

                let new_new_sims: Array2<f32> = new_union_data.dot(&new_union_data.t()); // Matlab line 139

                (
                    row,
                    new_row,
                    old_row,
                    new_row_union,
                    new_new_sims,
                    new_data,
                    old_data,
                )
            })
            .for_each(
                |(row, new_row, old_row, new_row_union, new_new_sims, new_data, old_data)| {
                    // Two for loops for the two distance tables (similarities and new_old_sims) for each pair of elements in the newNew list, their original ids
                    // First iterate over new_new_sims.. upper triangular (since distance table)

                    for new_ind1 in 0..new_row_union.len() - 1 {
                        // Matlab line 144 (-1 since don't want the diagonal)
                        let u1_id = new_row_union[new_ind1];

                        for new_ind2 in new_ind1 + 1..new_row_union.len() {
                            // Matlab line 147
                            let u2_id = new_row_union[new_ind2];
                            // then get their similarity from the matrix
                            let this_sim = new_new_sims[[new_ind1, new_ind2]];
                            // is the current similarity greater than the biggest distance
                            // in the row for u1_id? if it's not, then do nothing

                            if this_sim > minimum_in(&similarities.row(u1_id)) {
                                // Matlab line 154 // global_mins[u1_id]
                                // if it is, then u2_id actually can't already be there

                                updates.add(u1_id, u2_id, this_sim);
                            }

                            if minimum_in(&similarities.row(u2_id)) < this_sim {
                                // Matlab line 166 // was global_mins[u2_id]
                                updates.add(u2_id, u1_id, this_sim);
                            }
                        } // Matlab line 175
                    }

                    // nnw do the news vs the olds, no reverse links
                    // newOldSims = newData * oldData';

                    let new_old_sims = new_data.dot(&old_data.t());

                    // and do the same for each pair of elements in the new_row/old_row

                    for new_ind1 in 0..new_row.len() {
                        // Matlab line 183  // rectangular matrix - need to look at all
                        let u1_id = new_row[new_ind1];
                        for new_ind2 in 0..old_row.len() {
                            let u2_id = old_row[new_ind2]; // Matlab line 186
                                                           // then get their distance from the matrix
                            let this_sim = new_old_sims[[new_ind1, new_ind2]];
                            // is the current distance greater than the biggest distance
                            // in the row for u1_id? if it's not, then do nothing
                            if this_sim > minimum_in(&similarities.row(u1_id)) {
                                // Matlab line 191 // global_mins[u1_id]
                                // if it is, then u2Id actually can't already be there
                                updates.add(u1_id, u2_id, this_sim);
                            }

                            if this_sim > minimum_in(&similarities.row(u2_id)) {
                                // Matlab line 203 // was global_mins[u2_id]
                                updates.add(u2_id, u1_id, this_sim);
                            }
                        }
                    }
                },
            );

        // Now apply all the updates.

        work_done = updates
            .into_inner()
            .into_par_iter()
            .zip(neighbours.axis_iter_mut(Axis(0)).into_par_iter())
            .zip(similarities.axis_iter_mut(Axis(0)).into_par_iter())
            .zip(neighbour_is_new.axis_iter_mut(Axis(0)).into_par_iter())
            .enumerate()
            .map(
                |(
                    row_id,
                    (
                        ((updates, mut neighbours_row), mut similarities_row),
                        mut neighbour_is_new_row,
                    ),
                )| {
                    updates
                        .into_iter()
                        .map(|update| {
                            let this_sim = update.sim;
                            let new_index = update.index;
                            if !neighbours_row.iter().any(|x| *x == new_index) {
                                // Matlab line 204
                                let insert_pos = index_of_min(&similarities_row.view());
                                neighbours_row[insert_pos] = new_index;
                                similarities_row[insert_pos] = this_sim;
                                neighbour_is_new_row[insert_pos] = true;
                                // global_mins[row_id] = minimum_in(&similarities.row(row_id));  // TODO Matlab line 198 Do this later - don't keep global_mins - would it be faster??
                                true
                            } else {
                                false
                            }
                        })
                        .fold(false, |acc, x| acc | x) as usize // .any() but it won't short circuit so all updates are applied!
                },
            )
            .sum::<usize>();

        let after = Instant::now();
        log::debug!("Phase 3: {} ms", ((after - now).as_millis() as f64));
    }

    let final_time = Instant::now();
    log::debug!(
        "Overall time 3: {} ms",
        ((final_time - start_time).as_millis() as f64)
    );
}

//***** Utility functions *****

pub fn get_new_reverse_links_not_in_forward(
    neighbours: &&mut Array2<usize>,
    similarities: &&mut Array2<f32>,
    reverse_list_size: usize,
    new: &Array2<usize>,
) -> (Array2<usize>, Array2<f32>) {
    let now = Instant::now();
    // initialise old' and new'  Matlab line 90
    let num_neighbours = neighbours.ncols();
    let num_data = neighbours.nrows();
    // the reverse NN table  Matlab line 91
    let mut reverse: Array2<usize> = Array2::from_elem((num_data, reverse_list_size), usize::MAX);
    // all the distances from reverse NN table.
    let mut reverse_sims: Array2<f32> = Array2::from_elem((num_data, reverse_list_size), f32::MIN); // was -1.0f32
                                                                                                    // reverse_ptr - how many reverse pointers for each entry in the dataset
    let mut reverse_count = Array1::from_elem(num_data, 0);

    // loop over all current entries in neighbours; add that entry to each row in the
    // reverse list if that id is in the forward NNs
    // there is a limit to the number of reverse ids we will store, as these
    // are in a zipf distribution, so we will add the most similar only

    for row in 0..num_data {
        // Matlab line 97
        // all_ids are the forward links in the current id's row
        let all_ids = &neighbours.row(row); // Matlab line 98
                                            // so for each one of these (there are k...):
        for id in 0..num_neighbours {
            // Matlab line 99 (updated)
            // get the id
            let this_id = &all_ids[id];
            // and how similar it is to the current id
            let local_sim = similarities[[row, id]];

            let new_forward_links = new.row(*this_id);

            let forward_links_dont_contain_this = !new_forward_links.iter().any(|x| *x == row);

            // if the reverse list isn't full, we will just add this one
            // this adds to a priority queue and keeps track of max
            // We are trying to find a set of reverse near neighbours with the
            // biggest similarity of size reverse_list_size.
            // first find all the forward links containing the row

            if forward_links_dont_contain_this {
                if reverse_count[*this_id] < reverse_list_size {
                    // if the list is not full
                    // update the reverse pointer list and the similarities
                    reverse[[*this_id, reverse_count[*this_id]]] = row;
                    reverse_sims[[*this_id, reverse_count[*this_id]]] = local_sim; // pop that in too
                    reverse_count[*this_id] = reverse_count[*this_id] + 1; // increment the count
                } else {
                    // but it is, so we will only add it if it's more similar than another one already there

                    let (position, value) = min_index_and_value(&reverse_sims.row(*this_id)); // Matlab line 109
                    if value < local_sim {
                        // Matlab line 110  if the value in reverse_sims is less similar we over write
                        reverse[[*this_id, position]] = row; // replace the old min with the new sim value
                        reverse_sims[[*this_id, position]] = local_sim;
                    }
                }
            }
        }
    }

    let after = Instant::now();
    log::debug!("Phase 2: {} ms", ((after - now).as_millis() as f64));
    (reverse, reverse_sims)
}

// Same as function above without new parameter.
pub fn get_reverse_links_not_in_forward(
    neighbours: &Array2<usize>,
    similarities: &Array2<f32>,
    reverse_list_size: usize,
) -> (Array2<usize>, Array2<f32>) {
    // initialise old' and new'  Matlab line 90
    // the reverse NN table  Matlab line 91
    let num_neighbours = neighbours.ncols();
    let num_data = neighbours.nrows();
    let mut reverse: Array2<usize> = Array2::from_elem((num_data, reverse_list_size), usize::MAX);
    // all the distances from reverse NN table.
    let mut reverse_sims: Array2<f32> = Array2::from_elem((num_data, reverse_list_size), f32::MIN); // was -1.0f32
                                                                                                    // reverse_ptr - how many reverse pointers for each entry in the dataset
    let mut reverse_count = Array1::from_elem(num_data, 0);

    // loop over all current entries in neighbours; add that entry to each row in the
    // reverse list if that id is in the forward NNs
    // there is a limit to the number of reverse ids we will store, as these
    // are in a zipf distribution, so we will add the most similar only

    for row in 0..num_data {
        // Matlab line 97
        // all_ids are the forward links in the current id's row
        let neighbour_ids_current_row = &neighbours.row(row); // Matlab line 98
                                                              // so for each one of these (there are k...):
        for col in 0..num_neighbours {
            // Matlab line 99 (updated)
            // get the id
            let next_id_in_row = &neighbour_ids_current_row[col];
            // and how similar it is to the current id
            let next_sim_in_row = similarities[[row, col]];

            let neighbours_of_next_id_in_row = neighbours.row(*next_id_in_row);

            let neighbours_of_next_dont_contain_current_row =
                !neighbours_of_next_id_in_row.iter().any(|x| *x == row);

            log::debug!(
                "Row {} col {} next_id {} sim {} neighbours of next {} don't contain {} row {}",
                row,
                col,
                next_id_in_row,
                next_sim_in_row,
                neighbours_of_next_id_in_row,
                row,
                neighbours_of_next_dont_contain_current_row
            );

            // if the reverse list isn't full, we will just add this one
            // this adds to a priority queue and keeps track of max
            // We are trying to find a set of reverse near neighbours with the
            // biggest similarity of size reverse_list_size.
            // first find all the forward links containing the row

            if neighbours_of_next_dont_contain_current_row {
                log::debug!("count is {} ", reverse_count[*next_id_in_row]);
                if reverse_count[*next_id_in_row] < reverse_list_size {
                    // if the list is not full
                    // update the reverse pointer list and the similarities
                    log::debug!(
                        "Adding row {} refers to {} insert position {}",
                        row,
                        *next_id_in_row,
                        reverse_count[*next_id_in_row]
                    );
                    reverse[[*next_id_in_row, reverse_count[*next_id_in_row]]] = row;
                    reverse_sims[[*next_id_in_row, reverse_count[*next_id_in_row]]] =
                        next_sim_in_row; // pop that in too
                    reverse_count[*next_id_in_row] = reverse_count[*next_id_in_row] + 1;
                // increment the count
                } else {
                    // it is full, so we will only add it if it's more similar than another one already there
                    let (position, value) = min_index_and_value(&reverse_sims.row(*next_id_in_row)); // Matlab line 109
                    log::debug!(
                        "full min index in {} and value of row {} are {} {}",
                        &reverse_sims.row(*next_id_in_row),
                        row,
                        position,
                        value
                    );
                    if value < next_sim_in_row {
                        // Matlab line 110  if the value in reverse_sims is less similar we over write
                        log::debug!("overwriting");
                        reverse[[*next_id_in_row, position]] = row; // replace the old min with the new sim value
                        reverse_sims[[*next_id_in_row, position]] = next_sim_in_row;
                    }
                }
            }
        }
    }

    (reverse, reverse_sims)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_reverse_links() {
        let reverse_gt_2: Vec<usize> = vec![
            4,
            18446744073709551615, // 3 & 4 refer to 0 - but 3 in forward links
            0,
            18446744073709551615, // 0 refers to 1
            0,
            1, // 0, 1, 2, 3, 4 refer to 2 but 2,3,4 are in the forward => 0,1
            1,
            18446744073709551615, // 0,1,2,4 to 3 - but 0,2,4 in forward => 1
            1,
            18446744073709551615,
        ]; // 1, 2, 3 refer to 4  - but 2,3 in forward -> 1

        let forward_links: Vec<usize> = vec![
            1, 2, 3, // 0
            2, 3, 4, // 1
            2, 3, 4, // 2
            0, 2, 4, // 3
            0, 2, 3, // 4
        ];

        let forward_sims = vec![
            0.9, 0.9, 0.7, // 0
            0.7, 0.9, 0.9, // 1
            0.7, 0.9, 0.9, // 2
            0.9, 0.7, 0.6, // 3
            0.9, 0.9, 0.5, //4
        ];

        let mut forward_links: Array2<usize> =
            Array2::from_shape_vec((5, 3), forward_links).unwrap();
        let mut forward_sims: Array2<f32> = Array2::from_shape_vec((5, 3), forward_sims).unwrap();
        let mut gt_links: Array2<usize> = Array2::from_shape_vec((5, 2), reverse_gt_2).unwrap();

        let reverse_links =
            get_reverse_links_not_in_forward(&&mut forward_links, &&mut forward_sims, 2);

        log::debug!(
            "Reverse links: {:?} reverse sims: {:?}",
            reverse_links.0,
            reverse_links.1
        );

        assert_eq!(reverse_links.0, gt_links);
    }
}

fn get_selected_data(dao: Rc<Dao<Array1<f32>>>, dims: usize, old_row: &Vec<usize>) -> Array2<f32> {
    // let old_data =
    old_row
        .iter()
        .map(|&index| dao.get_datum(index)) // &Array1<f32>
        .flat_map(|value| value.iter()) // f32
        .copied()
        .collect::<Array<f32, Ix1>>()
        .to_shape((old_row.len(), dims))
        .unwrap()
        .to_owned()
}

fn fill_false(row: &mut ArrayViewMut1<bool>, selector: &ArrayView1<usize>) {
    for i in 0..selector.len() {
        row[selector[i]] = false;
    }
}

fn fill_selected(
    to_fill: &mut ArrayViewMut1<usize>,
    fill_from: &ArrayView1<usize>,
    selector: &ArrayView1<usize>,
) {
    for (i, &sel_index) in selector.iter().enumerate() {
        to_fill[i] = fill_from[sel_index];
    }
}

fn get_slice_using_selected(
    source: &ArrayView2<f32>,
    selectors: &ArrayView1<usize>,
    result_shape: [usize; 2],
) -> Array2<f32> {
    let mut sliced = Array2::uninit(result_shape);

    for count in 0..selectors.len() {
        // was result_shape
        source
            .slice(s![selectors[count], 0..])
            .assign_to(sliced.slice_mut(s![count, 0..]));
    }

    unsafe { sliced.assume_init() }
}

fn get_1D_slice_using_selected<T: Clone>(
    source: &ArrayView1<T>,
    selectors: &ArrayView1<usize>,
) -> Array1<T> {
    let mut sliced = Array1::uninit(selectors.len());

    for count in 0..selectors.len() {
        source
            .slice(s![selectors[count]])
            .assign_to(sliced.slice_mut(s![count]));
    }

    unsafe { sliced.assume_init() }
}

//************** BSP impl below here **************

pub fn initialise_table_bsp(
    dao: Rc<Dao<EvpBits<2>>>,
    chunk_size: usize,
    num_neighbours: usize,
) -> (Array2<usize>, Array2<f32>) {
    log::info!(
        "initializing table bsp, chunk_size: {chunk_size}, num_neighbours: {num_neighbours}"
    );

    let start_time = Instant::now();

    let num_data = dao.num_data;
    let dims = dao.get_dim();
    let data = dao.get_data();

    if num_neighbours > chunk_size {
        panic!(
            "Error: num_neighbours {} > chunk_size {}",
            num_neighbours, chunk_size
        );
    }
    let mut result_indices =
        unsafe { Array2::<usize>::uninit((num_data, num_neighbours)).assume_init() };
    let mut result_sims =
        unsafe { Array2::<f32>::uninit((num_data, num_neighbours)).assume_init() };

    log::info!(
        "result sizes: indices: {}, sims: {}",
        bytes_fmt(result_indices.len() * size_of::<usize>()),
        bytes_fmt(result_sims.len() * size_of::<f32>())
    );

    result_indices
        .axis_chunks_iter_mut(Axis(0), chunk_size)
        .into_iter() // .into_par_iter()
        .zip(result_sims.axis_chunks_iter_mut(Axis(0), chunk_size))
        .enumerate()
        .for_each(|(i, (mut result_indices_chunk, mut result_sims_chunk))| {
            let real_chunk_size = result_sims_chunk.shape()[0];
            let start_pos = i * chunk_size;
            let end_pos = start_pos + real_chunk_size;

            let chunk = data.slice(s![start_pos..end_pos]);

            let original_row_ids = rand_perm(num_data, real_chunk_size); // random data ids from whole data set
            let rand_data = get_slice_using_selectors(
                &data,
                &original_row_ids.view(),
                chunk.shape().try_into().unwrap(),
            ); // a view of the original data points as a matrix

            let chunk_dists: Array2<f32> =
                matrix_dot_bsp::<2>(&chunk, &rand_data.view(), |a, b| {
                    bsp_similarity_as_f32::<2>(a, b)
                }); // matrix mult all the distances - all relative to the original_rows

            let (sorted_ords, sorted_dists) = arg_sort_big_to_small_2d(&chunk_dists.view()); // sorted ords are row relative indices.
                                                                                             // these ords are row relative all range from 0..real_chunk_size

            // get the num_neighbours closest original data indices
            let mut closest_dao_indices: Array2<usize> =
                Array2::<usize>::zeros((real_chunk_size, num_neighbours));

            for row in 0..real_chunk_size {
                for col in 0..num_neighbours {
                    closest_dao_indices[[row, col]] = original_row_ids[sorted_ords[[row, col]]];
                }
            }

            result_indices_chunk.assign(&closest_dao_indices.slice(s![.., 0..num_neighbours]));
            result_sims_chunk.assign(&sorted_dists.slice(s![.., 0..num_neighbours]));
        });

    let end_time = Instant::now();
    log::debug!(
        "Initialistion in {:?}ms",
        ((end_time - start_time).as_millis() as f64)
    );

    // max bits is 64 * 4 * bits * 2 = 1024 + 200 =  1224 hardwire for now TODO parameterise
    (
        insert_index_at_position_1_inplace(result_indices),
        insert_column_inplace(result_sims, 1224.0),
    )
}

pub fn get_nn_table2_bsp(
    dao: Rc<Dao<EvpBits<2>>>,
    neighbours: &mut Array2<usize>,
    similarities: &mut Array2<f32>, // bigger is better
    num_neighbours: usize,
    rho: f64,
    delta: f64,
    reverse_list_size: usize,
) {
    log::warn!(
        "neighbours: {}, similarities: {}",
        bytes_fmt(neighbours.len() * size_of::<usize>()),
        bytes_fmt(similarities.len() * size_of::<f32>())
    );

    let start_time = Instant::now();

    let num_data = dao.num_data;
    let dims = dao.get_dim();
    let data = dao.get_data();

    // Matlab lines refer to richard_build.txt file in the matlab dir

    let mut iterations = 0;
    let mut neighbour_is_new = Array2::from_elem((num_data, num_neighbours), true);
    let mut work_done = num_data; // a count of the number of times a similarity minimum of row has changed - measure of flux

    while work_done > ((num_data as f64) * delta) as usize {
        // Matlab line 61
        // condition is fraction of lines whose min similarity has changed when this gets low - no much work done then stop.
        iterations += 1;

        log::debug!(
            "iterating: c: {} num_data: {} iters: {}",
            work_done,
            num_data,
            iterations
        );

        // phase 1

        let now = Instant::now();

        let mut new: Array2<usize> = Array2::from_elem((num_data, num_neighbours), 0); // Matlab line 65
        let mut old: Array2<usize> = Array2::from_elem((num_data, num_neighbours), 0);

        // initialise old and new inline

        for row in 0..num_data {
            // in Matlab line 74
            let row_flags = &neighbour_is_new.row_mut(row); // Matlab line 74

            // new_indices are the indices in this row whose flag is set to true (columns)

            let new_indices = row_flags // Matlab line 76
                .iter()
                .enumerate()
                .filter_map(|(index, flag)| if *flag { Some(index) } else { None })
                .collect::<Array1<usize>>();

            // old_indices are the indices in this row whose flag is set to false (intially there are none of these).

            let old_indices = row_flags // Matlab line 77
                .iter()
                .enumerate()
                .filter_map(|(index, flag)| if !*flag { Some(index) } else { None })
                .collect::<Array1<usize>>();

            // random data ids from whole data set
            // in matlab p = randperm(n,k) returns a row vector containing k unique integers selected randomly from 1 to n

            let sampled = rand_perm(
                new_indices.len(),
                (rho * (new_indices.len() as f64)).round() as u64 as usize,
            );

            // sampled are random indices from new_indices

            let mut new_row_view: ArrayViewMut1<usize> = new.row_mut(row);
            let mut neighbour_row_view: ArrayViewMut1<bool> = neighbour_is_new.row_mut(row);

            fill_selected(&mut new_row_view, &neighbours.row(row), &sampled.view()); // Matlab line 79
            fill_selected(&mut new_row_view, &neighbours.row(row), &old_indices.view());
            fill_false(&mut neighbour_row_view, &sampled.view())
        }

        let after = Instant::now();
        log::debug!("Phase 1: {} ms", ((after - now).as_millis() as f64));

        // phase 2  Matlab line 88

        let now = Instant::now();

        // initialise old' and new'  Matlab line 90

        // the reverse NN table  Matlab line 91
        let mut reverse: Array2<usize> = Array2::from_elem((num_data, reverse_list_size), 0);
        // all the distances from reverse NN table.
        let mut reverse_sims: Array2<f32> =
            Array2::from_elem((num_data, reverse_list_size), -1.0f32);
        // reverse_ptr - how many reverse pointers for each entry in the dataset
        let mut reverse_count = Array1::from_elem(num_data, 0);

        // loop over all current entries in neighbours; add that entry to each row in the
        // reverse list if that id is in the forward NNs
        // there is a limit to the number of reverse ids we will store, as these
        // are in a zipf distribution, so we will add the most similar only

        for row in 0..num_data {
            // Matlab line 97
            // all_ids are the forward links in the current id's row
            let all_ids = &neighbours.row(row); // Matlab line 98
                                                // so for each one of these (there are k...):
            for id in 0..num_neighbours {
                // Matlab line 99 (updated)
                // get the id
                let this_id = &all_ids[id];
                // and how similar it is to the current id
                let local_sim = similarities[[row, id]];

                // newForwardLinks = new(thisId,:);
                let new_forward_links = new.row(*this_id);

                // forwardLinksDontContainThis = sum(newForwardLinks == i_phase2) == 0;
                let forward_links_dont_contain_this = !new_forward_links.iter().any(|x| *x == row);

                // if the reverse list isn't full, we will just add this one
                // this adds to a priority queue and keeps track of max

                // We are trying to find a set of reverse near neighbours with the
                // biggest similarity of size reverse_list_size.
                // first find all the forward links containing the row

                if forward_links_dont_contain_this {
                    // if the reverse list isn't full, we will just add this one
                    // this adds to a priority queue and keeps track of max

                    // We are trying to find a set of reverse near neighbours with the
                    // biggest similarity of size reverse_list_size.
                    // first find all the forward links containing the row

                    if reverse_count[*this_id] < reverse_list_size {
                        // if the list is not full
                        // update the reverse pointer list and the similarities
                        reverse[[*this_id, reverse_count[*this_id]]] = row;
                        reverse_sims[[*this_id, reverse_count[*this_id]]] = local_sim; // pop that in too
                        reverse_count[*this_id] = reverse_count[*this_id] + 1; // increment the count
                    } else {
                        // but it is, so we will only add it if it's more similar than another one already there

                        let (position, value) = min_index_and_value(&reverse_sims.row(*this_id)); // Matlab line 109
                        if value < local_sim {
                            // Matlab line 110  if the value in reverse_sims is less similar we over write
                            reverse[[*this_id, position]] = row; // replace the old min with the new sim value
                            reverse_sims[[*this_id, position]] = local_sim;
                        }
                    }
                }
            }
        }

        let after = Instant::now();
        log::debug!("Phase 2: {} ms", ((after - now).as_millis() as f64));

        // phase 3

        let now = Instant::now();

        work_done = 0;

        let formatted_data: Vec<(
            usize,
            ArrayViewMut1<usize>,
            ArrayViewMut1<usize>,
            Array1<usize>,
            Array2<f32>,
            Array1<EvpBits<2>>,
            Array1<EvpBits<2>>,
        )> = old
            .axis_iter_mut(Axis(0)) // Get mutable rows (disjoint slices)
            .enumerate()
            .zip(new.axis_iter_mut(Axis(0)))
            .par_bridge()
            .map(|((row, old_row), new_row)| {
                let mut reverse_link_row: Array1<usize> = reverse
                    .row(row)
                    .iter()
                    .filter(|&&v| v != 0)
                    .map(|&x| x)
                    .collect::<Array1<usize>>();

                if rho < 1.0 {
                    // Matlab line 127
                    // randomly shorten the reverse_link_row vector
                    let reverse_indices = rand_perm(
                        reverse_link_row.len(),
                        (rho * reverse_link_row.len() as f64).round() as usize,
                    );
                    reverse_link_row = reverse_indices
                        .iter()
                        .map(|&i| reverse_link_row[i])
                        .collect::<Array1<usize>>();
                }
                let mut new_row_union: Array1<usize> = if new_row.len() == 0 {
                    // Matlab line 130
                    Array1::from(vec![])
                } else {
                    new_row
                        .iter()
                        .copied()
                        .chain(reverse_link_row.iter().copied())
                        .collect::<Array1<usize>>() // <<<<< 2 row copies here
                };

                let new_row_union_len = new_row_union.len();

                // index the data using the rows indicated in old_row
                let old_data = get_slice_using_selectors(&data, &old_row.view(), [old_row.len()]); // Matlab line 136
                let new_data = get_slice_using_selectors(&data, &new_row.view(), [new_row.len()]); // Matlab line 137
                let new_union_data =
                    get_slice_using_selectors(&data, &new_row_union.view(), [new_row_union_len]); // Matlab line 137

                let new_new_sims: Array2<f32> =
                    matrix_dot_bsp::<2>(&new_union_data.view(), &new_union_data.view(), |a, b| {
                        bsp_similarity_as_f32::<2>(a, b)
                    }); // <<<<<<<<< TODO CHECK!!

                (
                    row,
                    new_row,
                    old_row,
                    new_row_union,
                    new_new_sims,
                    new_data,
                    old_data,
                )
            })
            .collect();
        
        // SCHEMA: First bit denotes whether an update is done. Second bit denotes whether this item has been seen.
        let mut updates_done_this_iter: Vec<u8> = vec![0; num_data];
        let num_looked_at: AtomicUsize = AtomicUsize::new(0);
             
        // Checks that the last bit (whether it has been checked or not) is set to TRUE for all entries
        while num_looked_at.load(std::sync::atomic::Ordering::SeqCst) < num_data {

            let updates = Updates::new(num_data);
            let n_updates_to_apply = AtomicUsize::new(0);

            updates_done_this_iter = formatted_data
                .iter()
                .zip(updates_done_this_iter.clone())
                // Don't repeat the work!!
                .par_bridge()
                .map(
                    |((row, new_row, old_row, new_row_union, new_new_sims, new_data, old_data), row_update_state)| {
                        let mut new_row_update_state = row_update_state;
                        let ok_to_do_work = n_updates_to_apply                 
                            .load(std::sync::atomic::Ordering::SeqCst) < PARALLEL_WORK && (updates_done_this_iter[*row] & 0b1 as u8) == 0; // Checks whether this item is unseen
                        // Two for loops for the two distance tables (similarities and new_old_sims) for each pair of elements in the newNew list, their original ids
                        // First iterate over new_new_sims.. upper triangular (since distance table)

                        if ok_to_do_work {
                            new_row_update_state = new_row_update_state | 0b1;
                            num_looked_at.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

                            for new_ind1 in 0..new_row_union.len() - 1 {
                                // Matlab line 144 (-1 since don't want the diagonal)
                                let u1_id = new_row_union[new_ind1];

                                for new_ind2 in new_ind1 + 1..new_row_union.len() {
                                    // Matlab line 147
                                    let u2_id = new_row_union[new_ind2];
                                    // then get their similarity from the matrix
                                    let this_sim = new_new_sims[[new_ind1, new_ind2]];
                                    // is the current similarity greater than the biggest distance
                                    // in the row for u1_id? if it's not, then do nothing

                                    if this_sim > minimum_in(&similarities.row(u1_id)) {
                                        // Matlab line 154 // global_mins[u1_id]
                                        // if it is, then u2_id actually can't already be there
                                        updates.add(u1_id, u2_id, this_sim);
                                        n_updates_to_apply
                                            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                                    }

                                    if minimum_in(&similarities.row(u2_id)) < this_sim {
                                        // Matlab line 166 // was global_mins[u2_id]
                                        updates.add(u2_id, u1_id, this_sim);
                                        n_updates_to_apply
                                            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                                    }
                                } // Matlab line 175
                            }

                            // nnw do the news vs the olds, no reverse links
                            // newOldSims = newData * oldData';

                            let new_old_sims =
                                matrix_dot_bsp::<2>(&new_data.view(), &old_data.view(), |a, b| {
                                    bsp_similarity_as_f32::<2>(a, b)
                                });

                            // and do the same for each pair of elements in the new_row/old_row

                            for new_ind1 in 0..new_row.len() {
                                // Matlab line 183  // rectangular matrix - need to look at all
                                let u1_id = new_row[new_ind1];
                                for new_ind2 in 0..old_row.len() {
                                    let u2_id = old_row[new_ind2]; // Matlab line 186
                                                                   // then get their distance from the matrix
                                    let this_sim = new_old_sims[[new_ind1, new_ind2]];
                                    // is the current distance greater than the biggest distance
                                    // in the row for u1_id? if it's not, then do nothing

                                    if this_sim > minimum_in(&similarities.row(u1_id)) {
                                        // Matlab line 191 // global_mins[u1_id]
                                        // if it is, then u2Id actually can't already be there
                                        updates.add(u1_id, u2_id, this_sim);
                                        n_updates_to_apply
                                            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                                    }

                                    if this_sim > minimum_in(&similarities.row(u2_id)) {
                                        // Matlab line 203 // was global_mins[u2_id]
                                        updates.add(u2_id, u1_id, this_sim);
                                        n_updates_to_apply
                                            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                                    }
                                }
                            }
                        }
                        new_row_update_state
                    },
                ).collect::<Vec<u8>>();

            // println!("Updates to apply: {}", updates.len());

            // Now apply all the updates.
            let rows_updated: Vec<u8> = updates
                .into_inner()
                .into_par_iter()
                .zip(neighbours.axis_iter_mut(Axis(0)).into_par_iter())
                .zip(similarities.axis_iter_mut(Axis(0)).into_par_iter())
                .zip(neighbour_is_new.axis_iter_mut(Axis(0)).into_par_iter())
                .enumerate()
                .map(
                    |(
                        row_id,
                        (
                            ((updates, mut neighbours_row), mut similarities_row),
                            mut neighbour_is_new_row,
                        ),
                    )| {
                        updates
                            .into_iter()
                            .map(|update| {
                                let this_sim = update.sim;
                                let new_index = update.index;
                                if !neighbours_row.iter().any(|x| *x == new_index) {
                                    // Matlab line 204
                                    let insert_pos = index_of_min(&similarities_row.view());
                                    neighbours_row[insert_pos] = new_index;
                                    similarities_row[insert_pos] = this_sim;
                                    neighbour_is_new_row[insert_pos] = true;
                                    // global_mins[row_id] = minimum_in(&similarities.row(row_id));  // TODO Matlab line 198 Do this later. <<<<<<<<<<<<<<<<<<<<<<
                                    0b10 as u8
                                } else {
                                    0b00 as u8
                                }
                            })
                            .fold(0, |acc, x| if x > 0 {acc + 1} else {acc}) // .any() but it won't short circuit so all updates are applied!
                    },
                ).collect();

                // println!("rows_updated shape: {}", rows_updated.len());

                updates_done_this_iter = updates_done_this_iter.into_iter().zip(rows_updated).map(|x| x.0 | x.1).collect();
                work_done = updates_done_this_iter.iter().fold(0, |acc, x| acc + ((*x & 0b10) as usize));

                // println!("Work done: {}", work_done);
        }

        let after = Instant::now();
        log::debug!("Phase 3: {} ms", ((after - now).as_millis() as f64));
    }

    let final_time = Instant::now();
    log::debug!(
        "Overall time 3: {} ms",
        ((final_time - start_time).as_millis() as f64)
    );
}

//********* Helper functions *********

fn get_slice_using_selectors<T: Clone>(
    source: &ArrayView1<T>,
    selectors: &ArrayView1<usize>,
    result_shape: [usize; 1],
) -> Array1<T> {
    let mut sliced = Array1::uninit(result_shape); //

    for count in 0..selectors.len() {
        // was result_shape
        source
            .slice(s![selectors[count]])
            .assign_to(sliced.slice_mut(s![count]));
    }

    unsafe { sliced.assume_init() }
}

fn get_2D_slice_using<T: Clone>(
    source: &ArrayView2<T>,
    selectors: &ArrayView1<usize>,
) -> Array2<T> {
    let mut sliced = Array2::uninit((selectors.len(), source.ncols()));

    for count in 0..selectors.len() {
        // was result_shape
        source
            .slice(s![selectors[count], 0..])
            .assign_to(sliced.slice_mut(s![count, 0..]));
    }

    unsafe { sliced.assume_init() }
}

// inserts a new value (1 in this code) into the first column and moves rest of the values over
// use this for dists
fn insert_column_inplace(mut array: Array2<f32>, new_col_val: f32) -> Array2<f32> {
    let (nrows, ncols_plus_1) = array.dim();
    let ncols = ncols_plus_1 - 1;

    // SAFETY: Get a raw pointer to the data
    let data_ptr = array.as_mut_ptr();

    for row in (0..nrows).rev() {
        unsafe {
            // Move existing elements one slot right, starting from the end
            for col in (0..ncols).rev() {
                let src = data_ptr.add(row * ncols_plus_1 + col);
                let dst = data_ptr.add(row * ncols_plus_1 + col + 1);
                ptr::copy_nonoverlapping(src, dst, 1);
            }

            // Write the new value at the first column
            let first_col = data_ptr.add(row * ncols_plus_1);
            ptr::write(first_col, new_col_val);
        }
    }

    array
}

// inserts the index of the row into the first slot
// use this for ords
fn insert_index_at_position_1_inplace(mut array: Array2<usize>) -> Array2<usize> {
    let (nrows, ncols_plus_1) = array.dim();
    let ncols = ncols_plus_1 - 1;

    // SAFETY: Get a raw pointer to the data
    let data_ptr = array.as_mut_ptr();

    for row in (0..nrows).rev() {
        unsafe {
            // Move existing elements one slot right, starting from the end
            for col in (0..ncols).rev() {
                let src = data_ptr.add(row * ncols_plus_1 + col);
                let dst = data_ptr.add(row * ncols_plus_1 + col + 1);
                ptr::copy_nonoverlapping(src, dst, 1);
            }

            // Write the row index into the first column
            let first_col = data_ptr.add(row * ncols_plus_1);
            ptr::write(first_col, row);
        }
    }

    array
}
