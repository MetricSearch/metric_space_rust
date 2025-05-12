//! This implementation of Richard's NN table builder

mod updates;

use std::cmp::Ordering;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use ndarray::{s, Array, Array1, Array2, ArrayView, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Axis, CowArray, Dim, Ix, Ix1, Ix2, Order};
use ndarray::parallel::prelude::{IntoParallelIterator, IntoParallelRefIterator};
use dao::{Dao, DaoMatrix};
use rand_chacha::rand_core::SeedableRng;
use serde::{Deserialize, Serialize};
use utils::{arg_sort_big_to_small, arg_sort_small_to_big, index_of_min, min_index_and_value, minimum_in, rand_perm};
use utils::non_nan::NonNan;
use crate::updates::Updates;
use rayon::prelude::*;
use ndarray::parallel::prelude::*;


#[derive(Serialize, Deserialize)]
pub struct RDescentMatrix {
    pub indices: Array2<usize>,
    pub dists: Array2<f32>,
}

impl RDescentMatrix {
    pub fn new( dao: Rc<DaoMatrix>, num_neighbours: usize, reverse_list_size : usize, chunk_size : usize, rho: f64, delta : f64 ) -> RDescentMatrix {

        // let reverse_list_size = 64;
        // let rho: f64 = 1.0;
        // let delta = 0.01;
        // let chunk_size = 20000;
        let rng = rand_chacha::ChaCha8Rng::seed_from_u64(324 * 142); // random number
        let (mut ords, mut dists) = initialise_table_m(dao.clone(), chunk_size, num_neighbours);
        get_nn_table2(dao.clone(), &mut ords, &mut dists, num_neighbours, rho, delta, reverse_list_size);

        Self { indices: ords, dists: dists   }
    }
}

pub fn initialise_table_m(dao: Rc<DaoMatrix>, chunk_size: usize, num_neighbours: usize) -> (Array2<usize>,Array2<f32>) {

    let start_time = Instant::now();

    let num_data = dao.num_data;
    let dims = dao.get_dim();
    let data = dao.get_data();
    let num_loops = num_data / chunk_size;

    let mut result_indices = unsafe { Array2::<usize>::uninit((num_data, num_neighbours)).assume_init()};
    let mut result_sims = unsafe {Array2::<f32>::uninit((num_data, num_neighbours)).assume_init()};

    result_indices
        .axis_chunks_iter_mut(Axis(0),chunk_size)
        .into_par_iter()
        .zip(result_sims.axis_chunks_iter_mut(Axis(0),chunk_size)
            .into_par_iter())
        .enumerate()
        .for_each(|(i, (mut result_indices_chunk, mut result_sims_chunk))| {
            let start_pos = i * chunk_size;
            let end_pos = start_pos + chunk_size;

            let chunk = data.slice(s![start_pos..end_pos, 0..]);

            let rand_ids = rand_perm(num_data, chunk_size);  // random data ids from whole data set
            let rand_data = get_slice_using_selected(&data, &rand_ids.view(), chunk.shape().try_into().unwrap());

            let chunk_dists: Array2<f32> = rand_data.dot(&chunk.t()); // matrix mult all the distances

            let (sorted_ords, sorted_dists)= arg_sort_big_to_small(&chunk_dists);

            // get the num_neighbours closest original data indices

            let mut closest_dao_indices: Array2<usize> = Array2::<usize>::zeros((chunk_size, num_neighbours));

            for row in 0..sorted_ords.nrows() {
                for col in 0..num_neighbours {
                    closest_dao_indices[[row,col]] = rand_ids[sorted_ords[[row,col]]] as usize;
                }
            }



                result_indices_chunk
                    .assign(&closest_dao_indices.slice(s![.., 0..num_neighbours]));
                result_sims_chunk
                    .assign(&sorted_dists.slice(s![.., 0..num_neighbours]));


        });

    let end_time = Instant::now();
    println!("Initialistion in {:?}ms", ((end_time - start_time).as_millis() as f64) );

   (result_indices, result_sims)
}


pub fn get_nn_table2(dao: Rc<DaoMatrix>,
                     mut neighbours: &mut Array2<usize>,
                     mut similarities: &mut Array2<f32>, // bigger is better
                     num_neighbours: usize,
                     rho: f64, delta: f64, reverse_list_size: usize ) {

    let start_time = Instant::now();

    let num_data = dao.num_data;
    let dims = dao.get_dim();
    let data = dao.get_data();

    // Matlab lines refer to richard_build.txt file in the matlab dir

    // let mut global_mins = similarities // Matlab line 53
    //     .rows()
    //     .into_iter()
    //     .map(|row| {
    //         row
    //             .iter()
    //             .map(|f| NonNan(*f))
    //             .min().unwrap().0
    //     })
    //     .collect::<Array1<f32>>();

    let mut iterations = 0;
    let mut neighbour_is_new = Array2::from_elem((num_data, num_neighbours), true);
    let mut work_done = num_data; // a count of the number of times a similarity minimum of row has changed - measure of flux

    while work_done > (( num_data as f64 ) * delta ) as usize { // Matlab line 61
        // condition is fraction of lines whose min similarity has changed when this gets low - no much work done then stop.
        iterations += 1;

        println!("iterating: c: {} num_data: {} iters: {}", work_done, num_data, iterations);

        // phase 1

        let now = Instant::now();

        let mut new: Array2<usize> = Array2::from_elem((num_data, num_neighbours), 0); // Matlab line 65
        let mut old: Array2<usize> = Array2::from_elem((num_data, num_neighbours), 0);

        // initialise old and new inline

        for row in 0..num_data { // in Matlab line 74
            let row_flags = &neighbour_is_new.row_mut(row); // Matlab line 74

            // new_indices are the indices in this row whose flag is set to true (columns)

            let new_indices = row_flags // Matlab line 76
                .iter()
                .enumerate()
                .filter_map(|(index,flag)| { if *flag { Some(index) } else {None} } )
                .collect::<Array1<usize>>();

            // old_indices are the indices in this row whose flag is set to false (intially there are none of these).

            let old_indices = row_flags // Matlab line 77
                .iter()
                .enumerate()
                .filter_map(|(index,flag)| { if ! *flag  { Some(index) } else {None} } )
                .collect::<Array1<usize>>();

            // random data ids from whole data set
            // in matlab p = randperm(n,k) returns a row vector containing k unique integers selected randomly from 1 to n

            let sampled = rand_perm(new_indices.len(),(rho * (new_indices.len() as f64)).round() as u64 as usize);

            // sampled are random indices from new_indices

            let mut new_row_view: ArrayViewMut1<usize> = new.row_mut(row);
            let mut neighbour_row_view: ArrayViewMut1<bool> = neighbour_is_new.row_mut(row);

            fill_selected( &mut new_row_view, &neighbours.row(row), &sampled.view() );    // Matlab line 79
            fill_selected( &mut new_row_view,&neighbours.row(row), &old_indices.view() );
            fill_false( &mut neighbour_row_view, &sampled.view())
        }

        let after = Instant::now();
        println!("Phase 1: {} ms", ((after - now).as_millis() as f64) );

        // phase 2  Matlab line 88

        let now = Instant::now();

        // initialise old' and new'  Matlab line 90

        // the reverse NN table  Matlab line 91
        let mut reverse: Array2<usize> = Array2::from_elem((num_data, reverse_list_size), 0);
        // all the distances from reverse NN table.
        let mut reverse_sims: Array2<f32> = Array2::from_elem((num_data, reverse_list_size), -1.0f32);
        // reverse_ptr - how many reverse pointers for each entry in the dataset
        let mut reverse_count = Array1::from_elem(num_data,0);

        // loop over all current entries in neighbours; add that entry to each row in the
        // reverse list if that id is in the forward NNs
        // there is a limit to the number of reverse ids we will store, as these
        // are in a zipf distribution, so we will add the most similar only

        for row in 0..num_data { // Matlab line 97
            // all_ids are the forward links in the current id's row
            let all_ids = &neighbours.row(row); // Matlab line 98
            // so for each one of these (there are k...):
            for id in 0..num_neighbours { // Matlab line 99 (updated)
                // get the id
                let this_id = &all_ids[id];
                // and how similar it is to the current id
                let local_sim = similarities[[row,id]];

                // if the reverse list isn't full, we will just add this one
                // this adds to a priority queue and keeps track of max

                // We are trying to find a set of reverse near neighbours with the
                // biggest similarity of size reverse_list_size.
                // first find all the forward links containing the row

                if reverse_count[*this_id] < reverse_list_size { // if the list is not full
                    // update the reverse pointer list and the similarities
                    reverse[[*this_id,reverse_count[*this_id]]] = row;
                    reverse_sims[[*this_id,reverse_count[*this_id]]] = local_sim; // pop that in too
                    reverse_count[*this_id] = reverse_count[*this_id] + 1; // increment the count
                } else {
                    // but it is, so we will only add it if it's more similar than another one already there

                    let (position, value ) = min_index_and_value(&reverse_sims.row(*this_id)); // Matlab line 109
                    if value < local_sim { // Matlab line 110  if the value in reverse_sims is less similar we over write
                        reverse[[*this_id,position]] = row;  // replace the old min with the new sim value
                        reverse_sims[[*this_id,position]] = local_sim;
                    }
                }
            }
        }

        let after = Instant::now();
        println!("Phase 2: {} ms", ((after - now).as_millis() as f64) );

        // phase 3

        let now = Instant::now();

        work_done = 0;

        let mut updates = Updates::new(num_data);


        old
            .axis_iter_mut(Axis(0)) // Get mutable rows (disjoint slices)
            .enumerate()
            .zip( new.axis_iter_mut(Axis(0)) )
            .par_bridge()
            .map( |((row,old_row), new_row)| {

                let mut reverse_link_row: Array1<usize> = reverse.row(row).iter().filter(|&&v| v != 0).map(|&x| x).collect::<Array1<usize>>();

                if rho < 1.0 { // Matlab line 127
                    // randomly shorten the reverse_link_row vector
                    let reverse_indices = rand_perm(reverse_link_row.len(), (rho * reverse_link_row.len() as f64).round() as usize);
                    reverse_link_row = reverse_indices.iter().map(|&i| reverse_link_row[i]).collect::<Array1<usize>>();
                }
                let mut new_row_union: Array1<usize> = if new_row.len() == 0 {     // Matlab line 130
                    Array1::from(vec![])
                } else {
                    new_row.iter().copied().chain(reverse_link_row.iter().copied()).collect::<Array1<usize>>()   // <<<<< 2 row copies here
                };

                // let new_row_union: ArrayViewMut1<usize> = new_row_union.view_mut();

                let new_row_union_len = new_row_union.len();

                // index the data using the rows indicated in old_row
                let old_data = get_slice_using_selected(&data, &old_row.view(), [old_row.len(), dims]);                         // Matlab line 136
                let new_data = get_slice_using_selected(&data, &new_row.view(), [new_row.len(), dims]);                          // Matlab line 137
                let new_union_data = get_slice_using_selected(&data, &new_row_union.view(), [new_row_union_len, dims]);      // Matlab line 137

                let new_new_sims: Array2<f32> = new_union_data.dot(&new_union_data.t()); // Matlab line 139

                (row,new_row,old_row,new_row_union,new_new_sims,new_data,old_data)
            } )
            .for_each( |(row,new_row,old_row,new_row_union,new_new_sims,new_data,old_data)| {


            // if row == 1 {
            //     println!("neighbours[1]: {:?}", neighbours.row(1));
            //     println!("similarities[1]: {:?}", similarities.row(1));
            //     println!("new_row[1]: {:?}", new_row);
            //     println!("reverse_link_row[1]: {:?}", reverse_link_row);
            //     println!("new_row_union[1]: {:?}", new_row_union);
            //     println!("min_sims[1]): {:?}", global_mins[1]);
            //     println!("reverse[1]: {:?}", reverse.row(1));
            // }

            // Two for loops for the two distance tables (similarities and new_old_sims) for each pair of elements in the newNew list, their original ids

            // First iterate over new_new_sims.. upper triangular (since distance table)

            for new_ind1 in 0 .. new_row_union.len() - 1  { // Matlab line 144 (-1 since don't want the diagonal)
                let u1_id = new_row_union[new_ind1];

                for new_ind2 in new_ind1 + 1 .. new_row_union.len() { // Matlab line 147
                    let u2_id = new_row_union[new_ind2];
                    // then get their similarity from the matrix
                    let this_sim = new_new_sims[[new_ind1, new_ind2]];
                    // is the current similarity greater than the biggest distance
                    // in the row for u1_id? if it's not, then do nothing

                    if this_sim > minimum_in(&similarities.row(u1_id))     { // Matlab line 154 // global_mins[u1_id]
                        // if it is, then u2_id actually can't already be there

                        updates.add(u1_id, u2_id, this_sim);

                        // if ! neighbours.row(u1_id).iter().any(|x| *x == u2_id) { // Matlab line 156
                        //     // THIS IS LINE 157 of the text that is in richard_build_nns.txt (in matlab folder) and also below..
                        //     let position = index_of_min(&similarities.row(u1_id)); // Matlab line 157
                        //     neighbours[[u1_id,position]] = u2_id;
                        //     neighbour_is_new[[u1_id,position]] = true;
                        //     similarities[[u1_id,position]] = this_sim;
                        //     global_mins[u1_id] = minimum_in(&similarities.row(u1_id));
                        //     work_done = work_done + 1;
                        // }
                    }

                    if minimum_in(&similarities.row(u2_id)) < this_sim { // Matlab line 166 // was global_mins[u2_id]

                        updates.add(u2_id, u1_id, this_sim);
                        // if ! neighbours.row(u2_id).iter().any(|x| *x == u1_id) {
                        //     let position = index_of_min(&similarities.row(u2_id));
                        //     neighbours[[u2_id,position]] = u1_id;
                        //     neighbour_is_new[[u2_id,position]] = true;
                        //     similarities[[u2_id,position]] = this_sim;
                        //     global_mins[u2_id] = minimum_in(&similarities.row(u2_id));
                        //     work_done = work_done + 1;
                        // }
                    }
                } // Matlab line 175
            }

            // nnw do the news vs the olds, no reverse links
            // newOldSims = newData * oldData';

            let new_old_sims = new_data.dot(&old_data.t());

            // and do the same for each pair of elements in the new_row/old_row

            for new_ind1 in 0 .. new_row.len() { // Matlab line 183  // rectangular matrix - need to look at all
                let u1_id = new_row[new_ind1];
                for new_ind2 in 0..old_row.len() {
                    let u2_id = old_row[new_ind2]; // Matlab line 186
                    // then get their distance from the matrix
                    let this_sim = new_old_sims[[new_ind1,new_ind2]];
                    // is the current distance greater than the biggest distance
                    // in the row for u1_id? if it's not, then do nothing

                    if this_sim > minimum_in(&similarities.row(u1_id))  { // Matlab line 191 // global_mins[u1_id]
                        // if it is, then u2Id actually can't already be there

                        updates.add(u1_id, u2_id, this_sim);

                        // if ! neighbours.row(u1_id).iter().any(|x| *x == u2_id) { // Matlab line 193
                        //     let position = index_of_min(&similarities.row(u1_id));
                        //     neighbours[[u1_id,position]] = u2_id;
                        //     similarities[[u1_id,position]] = this_sim;
                        //     neighbour_is_new[[u1_id,position]] = true;
                        //     global_mins[u1_id] = minimum_in(&similarities.row(u1_id));  // Matlab line 198
                        //     work_done = work_done + 1;
                        // }
                    }

                    if this_sim > minimum_in(&similarities.row(u2_id)) { // Matlab line 203 // was global_mins[u2_id]

                        updates.add(u2_id, u1_id, this_sim);

                        // if ! neighbours.row(u2_id).iter().any(|x| *x == u1_id) { // Matlab line 204
                        //     let position = index_of_min(&similarities.row(u2_id));
                        //     neighbours[[u2_id,position]] = u1_id;
                        //     similarities[[u2_id,position]] = this_sim;
                        //     neighbour_is_new[[u2_id,position]] = true;
                        //     global_mins[u2_id] = minimum_in(&similarities.row(u2_id));
                        //     work_done = work_done + 1;  // Matlab line 210
                        // }
                    }
                }
            }
        } );

        // Now apply all the updates.

        work_done = updates
            .into_inner()
            .into_par_iter()
            .zip( neighbours.axis_iter_mut(Axis(0)).into_par_iter() )
            .zip( similarities.axis_iter_mut(Axis(0)).into_par_iter() )
            .zip( neighbour_is_new.axis_iter_mut(Axis(0)).into_par_iter() )
            .enumerate()
            .map( |(row_id,(((updates,mut neighbours_row),mut similarities_row),mut neighbour_is_new_row)) | {

                updates
                    .into_iter()
                    .map( |update| {
                        let this_sim = update.sim;
                        let new_index = update.index;
                        if ! neighbours_row.iter().any(|x| *x == new_index) { // Matlab line 204
                            let insert_pos = index_of_min(&similarities_row.view());
                            neighbours_row[ insert_pos] = new_index;
                            similarities_row[insert_pos] = this_sim;
                            neighbour_is_new_row[insert_pos] = true;
                            // global_mins[row_id] = minimum_in(&similarities.row(row_id));  // Matlab line 198 Do this later.
                            true
                        } else {
                            false
                        }
                    } )
                    .fold(false, |acc, x| acc | x) as usize // .any() but it won't short circuit so all updates are applied!

            }).sum::<usize>() ;



        let after = Instant::now();
        println!("Phase 3: {} ms", ((after - now).as_millis() as f64) );

        // println!("Min sums: {} min: {}", min_sums, overall_min);
        // println!( "Sims line 0: {:?} min {}", similarities.row(0),global_mins[0] );
    }

    let final_time = Instant::now();
    println!("Overall time 3: {} ms", ((final_time - start_time).as_millis() as f64) );
}

//***** Utility functions *****

fn get_selected_data(dao: Rc<Dao<Array1<f32>>>, dims: usize, old_row: &Vec<usize>) -> Array2<f32> {
    // let old_data =
        old_row
            .iter()
            .map(|&index| dao.get_datum(index)) // &Array1<f32>
            .map(|value| value.iter()) // f32
            .flatten()
            .map(|&value| value as f32)
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

fn fill_selected(to_fill: &mut ArrayViewMut1<usize>, fill_from: &ArrayView1<usize>, selector: &ArrayView1<usize>) {
    for (i, &sel_index) in selector.iter().enumerate() {
        to_fill[i] = fill_from[sel_index];
    }
}

fn get_slice_using_selected(from: &ArrayView2<f32>, selectors: &ArrayView1<usize>, result_shape: [usize; 2]) -> Array2<f32> {
    let mut sliced = Array2::uninit(result_shape); //

    for count in 0..result_shape[0] {
        from.slice(s![selectors[count],0..]).assign_to(sliced.slice_mut(s![count,0..]));
    }

    unsafe {
        sliced.assume_init()
    }
}

