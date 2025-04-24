//! This implementation of Richard's NN table builder

use std::cmp::Ordering;
use std::rc::Rc;
use std::time::Instant;
use ndarray::{s, Array, Array1, Array2, ArrayView2, Axis, CowArray, Dim, Ix, Ix1, Ix2, Order};
use dao::{Dao,DaoMatrix};
use rand_chacha::rand_core::SeedableRng;
use randperm_crt::{Permutation, RandomPermutation};
use serde::{Deserialize, Serialize};
use utils::{arg_sort_small_to_big, index_of_min, min_index_and_value, minimum_in};
use utils::non_nan::NonNan;

#[derive(Serialize, Deserialize)]
pub struct RDescentMatrix {
    pub indices: Vec<Vec<usize>>,
    pub dists: Vec<Vec<f32>>,
}

impl RDescentMatrix {
    pub fn new( dao: Rc<DaoMatrix>, num_neighbours: usize) -> RDescentMatrix {

        let reverse_list_size = 64;
        let rho: f64 = 1.0;
        let delta = 0.01;
        let chunk_size = 20000;
        let rng = rand_chacha::ChaCha8Rng::seed_from_u64(324 * 142); // random number
        let (ords,dists) = initialise_table_m(dao.clone(),chunk_size,num_neighbours);

        get_nn_table2_m(
            // mut current_graph,
            // dao,
            // &mut rng,
            // num_neighbours,
            // reverse_list_size,
            // rho,
            // delta,
        );
        Self { indices: ords,
            dists: dists   }
    }
}

// data, B, nnSims, k, rho, delta,
fn get_nn_table2_m(
    // current_graph: &mut Heap,
    // dao: Rc<Dao<Array1<f32>>>,
    // rng: &mut ChaCha8Rng,
    // num_neighbours: usize, // reverseListSize
    // reverse_list_size: usize,
    // rho: usize,
    // delta: f64,
) {
    todo!();
}

/*
    randperm(n,k) returns a vector containing k unique integers selected randomly from 1 to n.
*/
pub fn rand_perm(drawn_from: usize, how_many: usize ) -> Vec<usize> {
    if drawn_from == 0 {
        return Vec::new();
    }
    let perm = RandomPermutation::new(drawn_from as u64).unwrap();
    perm.iter().take(how_many).map(|x| x as usize).collect::<Vec<usize>>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_rnd_perm1() {
        let mut x = rand_perm(10,10);
        x.sort();
        assert_eq!(x.len(), 10);
        assert_eq!(x[0],0);
        assert_eq!(x[5],5);
        assert_eq!(x[9],9);
    }
    #[test]
    pub fn test_rnd_perm2() {
        let mut y = rand_perm(10,5);
        assert_eq!(y.len(), 5);
        assert!( y.iter().all(|&x| x >= 0 && x < 10 ) );

        y.sort();

        for i in 0..4 {
            assert!( y[i] < y[i+1] );
        }
    }
}

pub fn initialise_table_m(dao: Rc<DaoMatrix>, chunk_size: usize, num_neighbours: usize) -> (Array2<usize>>,Array2<f32>>) {
    let num_data = dao.num_data;
    let dims = dao.get_dim();
    let data = dao.get_data();
    let num_loops = num_data / chunk_size;

    let mut result_indices = Array2::<f64>::zeros((num_data, num_neighbours));
    let mut result_sims = Array2::<f32>::from_elem((num_data, num_neighbours), f32::MAX);

    for i in 0..num_loops {
        let start_pos = i * chunk_size;
        let end_pos = num_data.min(start_pos + chunk_size);
        let chunk = &data.slice(s![start_pos..end_pos, 0..]);  //select(Axis(0),&(start_pos..end_pos).collect::<Vec<usize>>());

        let rand_ids = rand_perm(num_data, chunk_size);  // random data ids from whole data set
        let rand_data = get_slice_using_selected(&data, rand_ids, chunk.shape());

        let chunk_transpose = &chunk.t();

        let chunk_dists: Array2<f32> = rand_data.dot(&chunk_transpose); // matrix mult all the distances

        // AL IS HERE

        let x = arg_sort_array2(&mut result_indices);
        let (sorted_ords, sorted_dists) = arg_sort_matrix(chunk_dists); // largest first, relative to to rand_data

        // get the num_neighbours closest original data indices

        let mut closest_dao_indices: Vec<Vec<usize>> = vec![vec![0; num_neighbours]; chunk_size];

        for row_index in 0..sorted_ords.len() {
            let row = &sorted_ords[row_index];
            for col in 0..num_neighbours {
                closest_dao_indices[row_index][col] = rand_ids[sorted_ords[row_index][col]] as usize;
            }
        }

        // Assign the nearest distances and neighbours to the result vectors
        for index in 0..sorted_dists.len() {
            result_indices[start_pos + index] = closest_dao_indices[index].to_vec();
            result_sims[start_pos + index] = sorted_dists[index][0..num_neighbours].to_vec();
        }
    }
    (result_indices, result_sims)
}


pub fn get_nn_table2(dao: Rc<Dao<Array1<f32>>>,
                     mut neighbours: &mut Vec<Vec<usize>>,
                     mut similarities: &mut Vec<Vec<f32>>, // bigger is better
                     num_neighbours: usize,
                     rho: f64, delta: f64, reverse_list_size: usize ) -> (Vec<Vec<usize>>,Vec<Vec<f32>>) {
    let start_time = Instant::now();

    let num_data = dao.num_data;
    let dims = dao.get_dim();

    // TODO change the DAO to matrix

    // Matlab lines refer to richard_build.txt file in the matlab dir

    let mut global_mins = similarities // Matlab line 53
        .iter()
        .map(|row| {
            row
                .iter()
                .map(|f| NonNan(*f))
                .min().unwrap().0
        })
        .collect::<Vec<f32>>(); // TODO should be an array1

    let mut iterations = 0;
    let mut neighbour_is_new: Vec<Vec<bool>> = vec![vec![true; num_neighbours]; num_data]; // TODO these should be contiguous
    let mut c = num_data; // a count of the number of times a similarity minimum of row has changed - measure of flux

    while c > ((num_data as f64) * delta) as usize { // Matlab line 61
        // condition is fraction of lines whose min similarity has changed
        // when this gets low - no much work done then stop.
        iterations += 1;

        println!("iterating: c: {} num_data: {} iters: {}", c, num_data, iterations);

        // phase 1

        println!("phase 1..");

        let now = Instant::now();

        // TODO these should be contiguous arrays...

        let mut new: Vec<Vec<usize>> = vec![vec![0; num_neighbours]; num_data]; // Matlab line 65
        let mut old: Vec<Vec<usize>> = vec![vec![0; num_neighbours]; num_data];

        // initialise old and new inline

        for row in 0..num_data { // in Matlab line 74
            let row_flags = &neighbour_is_new[row]; // Matlab line 74

            // new_indices are the indices in this row whose flag is set to true (columns)

            let new_indices = row_flags // Matlab line 76
                .iter()
                .enumerate()
                .filter_map(|(index, flag)| { if *flag { Some(index) } else { None } })
                .collect::<Vec<usize>>();

            // old_indices are the indices in this row whose flag is set to false (intially there are none of these).

            let old_indices = row_flags // Matlab line 77
                .iter()
                .enumerate()
                .filter_map(|(index, flag)| { if !*flag { Some(index) } else { None } })
                .collect::<Vec<usize>>();

            // random data ids from whole data set
            // in matlab p = randperm(n,k) returns a row vector containing k unique integers selected randomly from 1 to n

            let sampled = rand_perm(new_indices.len(), (rho * (new_indices.len() as f64)).round() as u64 as usize);

            // sampled are random indices from new_indices

            fill_selected(&mut new[row], &neighbours[row], &sampled);    // Matlab line 79
            fill_selected(&mut new[row], &neighbours[row], &old_indices);
            fill_false(&mut neighbour_is_new[row], &sampled)
        }

        let after = Instant::now();
        println!("Phase 1: {} ms", ((after - now).as_millis() as f64));

        // phase 2  Matlab line 88

        println!("phase 2..");

        let now = Instant::now();

        // initialise old' and new'  Matlab line 90

        // TODO these should be contiguous arrays...

        // the reverse NN table  Matlab line 91
        let mut reverse: Vec<Vec<usize>> = vec![vec![0; reverse_list_size]; num_data];
        // all the distances from reverse NN table.
        let mut reverse_sims = vec![vec![-1.0f32; reverse_list_size]; num_data];
        // reverse_ptr - how many reverse pointers for each entry in the dataset
        let mut reverse_ptr = vec![0; num_data];

        // loop over all current entries in neighbours; add that entry to each row in the
        // reverse list if that id is in the forward NNs
        // there is a limit to the number of reverse ids we will store, as these
        // are in a zipf distribution, so we will add the most similar only

        for row in 0..num_data { // Matlab line 97
            // all_ids are the forward links in the current id's row
            let all_ids = &neighbours[row]; // Matlab line 98
            // so for each one of these (there are k...):
            for id in 0..num_neighbours { // Matlab line 99 (updated)
                // get the id
                let this_id = &all_ids[id];
                // and how similar it is to the current id
                let local_sim = similarities[row][id];

                // if the reverse list isn't full, we will just add this one
                // this adds to a priority queue and keeps track of max

                /*
                    TODO check this:
                    Richard e-mail:

                    %if that reverse list isn't full, we will just add this one
                    nextReverseBlocation = reverseBptr(thisId) + 1;
                    if reverseBptr(thisId) <= reverseListSize % this should be reverseListSize!
                           reverseBptr(thisId) = nextReverseBlocation;
                           reverseB(thisId,nextReverseBlocation) = i_phase2;
                           reverseBsims(thisId,nextReverseBlocation) = localSim;
                    else
                 */

                // This code works other variations give index errors.

                let next_reverse_location = reverse_ptr[*this_id]; // Matlab line 102 // THIS IS +1 in matlab ******** RICHARD ?????

                if next_reverse_location < reverse_list_size { // al changed *** //<<<<<<<<<*****
                    reverse_ptr[*this_id] = next_reverse_location; // BEN + 1; // Richard bug - +1 wasn't there.
                    reverse[*this_id][next_reverse_location] = row;
                    reverse_sims[*this_id][next_reverse_location] = local_sim;
                } else {
                    // but it is, so we will only add it if it's more similar
                    // than another one already there

                    let (position, value) = min_index_and_value(&reverse_sims[*this_id]); // Matlab line 109

                    if value < local_sim { // Matlab line 110  if the value in reverse_sims is less similar we over write
                        reverse[*this_id][position] = row;
                        reverse_sims[*this_id][position] = local_sim;
                    }
                }
            }
        }

        let after = Instant::now();
        println!("Phase 2: {} ms", ((after - now).as_millis() as f64));

        // phase 3

        println!("phase 3..");
        let now = Instant::now();

        c = 0;
        for row in 0..num_data { // Matlab line 123

            // TODO these should be ArrayView

            let old_row: Vec<usize> = old[row].iter().filter(|&&v| v != 0).map(|&x| x).collect::<Vec<usize>>();
            let new_row: Vec<usize> = new[row].iter().filter(|&&v| v != 0).map(|&x| x).collect::<Vec<usize>>();
            let mut prime_row: Vec<usize> = reverse[row].iter().filter(|&&v| v != 0).map(|&x| x).collect::<Vec<usize>>();

            if rho < 1.0 { // Matlab line 127
                // randomly shorten the prime_row vector
                prime_row = rand_perm(prime_row.len(), (rho * prime_row.len() as f64).round() as usize);
            }
            // Matlab line 130
            let new_row_union: Vec<usize> = new_row.iter().copied().chain(prime_row.iter().copied()).collect::<Vec<usize>>();

            // ***** This seems to need lots of copying of the data  - once to make new_union_data and again in vecvec_to_ndarray.
            // TODO Can you do more with slices?

            // index the data using the rows indicated in old_row

            let old_data = get_selected_data(dao.clone(), dims, &old_row); // Matlab line 136
            let new_data = get_selected_data(dao.clone(), dims, &new_row); // Matlab line 137
            let new_union_data = get_selected_data(dao.clone(), dims, &new_row_union); // Matlab line 137
            let new_new_sims: Array2<f32> = new_union_data.dot(&new_union_data.t()); // Matlab line 139

            // separate for loops for the two distance tables...
            // for each pair of elements in the newNew list, their original ids

            for new_ind1 in 0..new_row_union.len() { // Matlab line 144
                let u1_id = new_row_union[new_ind1];

                for new_ind2 in new_ind1 + 1..new_row_union.len() { // Matlab line 147
                    let u2_id = new_row_union[new_ind2];
                    // then get their similarity from the matrix
                    let this_sim = new_new_sims[[new_ind1, new_ind2]];
                    // is the current similarity greater than the biggest distance
                    // in the row for u1_id? if it's not, then do nothing

                    if this_sim > global_mins[u1_id] { // Matlab line 154
                        // if it is, then u2_id actually can't already be there
                        if !neighbours[u1_id].iter().any(|x| *x == u2_id) { // Matlab line 156
                            // THIS IS LINE 157 of the text that is in richard_build_nns.txt (in matlab folder) and also below..
                            let wh = index_of_min(&similarities[u1_id]); // Matlab line 157
                            neighbours[u1_id][wh] = u2_id;
                            neighbour_is_new[u1_id][wh] = true;
                            similarities[u1_id][wh] = this_sim;
                            //println!( "1 Updating similarities {} {} {} ", u1_id, wh, this_sim );
                            global_mins[u1_id] = minimum_in(&similarities[u1_id]);
                            c = c + 1;
                        }
                    }

                    if global_mins[u2_id] < this_sim { // Matlab line 166
                        if !neighbours[u2_id].iter().any(|x| *x == u1_id) {
                            let wh = index_of_min(&similarities[u2_id]);
                            neighbours[u2_id][wh] = u1_id;
                            neighbour_is_new[u2_id][wh] = true;
                            similarities[u2_id][wh] = this_sim;
                            //println!( "2 Updating similarities {} {} {} ", u2_id, wh, this_sim );
                            global_mins[u2_id] = minimum_in(&similarities[u2_id]);
                            c = c + 1;
                        }
                    }
                } // Matlab line 175
            }

            // now do the news vs the olds, no reverse links
            // TODO check the amount of copying!

            let new_old_sims = new_data.dot(&old_data.t());

            // and do the same for each pair of elements in the new_row/old_row

            for new_ind1 in 0..new_row.len() { // Matlab line 183
                let u1_id = new_row[new_ind1];
                for new_ind2 in 0..old_row.len() {
                    let u2_id = old_row[new_ind2]; // Matlab line 186
                    // then get their distance from the matrix
                    let this_sim = new_old_sims.get((new_ind1, new_ind2)).unwrap().clone();
                    // is the current distance greater than the biggest distance
                    // in the row for u1Id? if it's not, then do nothing

                    if this_sim > global_mins[u1_id] { // Matlab line 191
                        // if it is, then u2Id actually can't already be there
                        if !neighbours[u1_id].iter().any(|x| *x == u2_id) { // Matlab line 193
                            let wh = index_of_min(&similarities[u1_id]);
                            neighbours[u1_id][wh] = u2_id;
                            similarities[u1_id][wh] = this_sim;
                            //println!( "3 Updating similarities {} {} {} ", u1_id, wh, this_sim );
                            neighbour_is_new[u1_id][wh] = true;
                            global_mins[u1_id] = minimum_in(&similarities[u1_id]);
                            c = c + 1;
                        }
                    }


                    if this_sim > global_mins[u2_id] { // Matlab line 203
                        if !neighbours[u2_id].iter().any(|x| *x == u1_id) { // Matlab line 204
                            let wh = index_of_min(&similarities[u2_id]);
                            neighbours[u2_id][wh] = u1_id;
                            similarities[u2_id][wh] = this_sim;
                            //println!( "4 Updating similarities {} {} {} ", u2_id, wh, this_sim );
                            neighbour_is_new[u2_id][wh] = true;
                            global_mins[u2_id] = minimum_in(&similarities[u2_id]);
                            c = c + 1;  // Matlab line 210
                        }
                    }
                }
            }
        }
        let after = Instant::now();
        println!("Phase 3: {} ms", ((after - now).as_millis() as f64));
        println!("c: {}, termination threshold: {}", c, ((num_data as f64) * delta) as usize);
    }

    let final_time = Instant::now();
    println!("Overall time 3: {} ms", ((final_time - start_time).as_millis() as f64));

    (neighbours.to_owned(), similarities.to_owned()) // TODO Does this copy?
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

fn fill_false(row: &mut Vec<bool>, selector: &Vec<usize>) {
    for i in 0..selector.len() {
        row[selector[i]] = false;
    }
}

fn fill_selected(to_fill: &mut Vec<usize>, fill_from: &Vec<usize>, selector: &Vec<usize> ) {
    for i in 0..selector.len() {
        to_fill[i] = fill_from[selector[i]];
    }
}

fn get_slice_using_selected(from: &ArrayView2<f32>, selectors: Vec<usize>, result_shape: &[usize]) -> Array2<f32> {
    let mut sliced = Array2::uninit(result_shape);

    for count in 0..result_shape[0] {
        from.slice(s![selectors[count],0..]).assign_to(sliced.slice_mut(s![count,0..]));
    }

    unsafe {
        sliced.assume_init()
    }
}
