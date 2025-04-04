//! This implementation of Richard's NN table builder

use std::cmp::Ordering;
use std::rc::Rc;
use ndarray::{Array, Array1, Array2, Axis, IndexLonger, Ix1, Order};
use dao::{Dao};
use rand_chacha::rand_core::SeedableRng;
use randperm_crt::{Permutation, RandomPermutation};
use serde::{Deserialize, Serialize};
use utils::arg_sort_array2;
use utils::non_nan::NonNan;

#[derive(Serialize, Deserialize)]
pub struct RDescent {
    pub indices: Vec<Vec<usize>>,
    pub dists: Vec<Vec<f32>>,
}

impl RDescent {
    pub fn new( dao: Rc<Dao<Array1<f32>>>, num_neighbours: usize) -> RDescent {

        let reverse_list_size = 64;
        let rho: f64 = 1.0;
        let delta = 0.01;
        let chunk_size = 20000;
        let rng = rand_chacha::ChaCha8Rng::seed_from_u64(324 * 142); // random number
        let (ords,dists) = initialise_table(dao.clone(),chunk_size,num_neighbours);

        get_nn_table2(
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
fn get_nn_table2(
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

// TODO use rand_perm is code below!
/*
    randperm(n,k) returns a vector containing k unique integers selected randomly from 1 to n.
*/
pub fn rand_perm(drawn_from: usize, how_many: usize ) -> Vec<usize> {
    let perm = RandomPermutation::new(drawn_from as u64).unwrap();
    perm.iter().take(how_many).map(|x| x as usize).collect::<Vec<usize>>()
}
pub fn vecvec_to_ndarray<T>(v: Vec<Vec<T>>) -> Array2<T> {
    let rows = v.len();
    let columns = v[0].len();
    let mut array = Array2::<T>::default((rows, columns));
    for (i, mut row) in array.axis_iter_mut(Axis(0)).enumerate() {
        for (j, col) in row.iter_mut().enumerate() {
            *col = v[i][j];
        }
    }
    array
}

pub fn initialise_table(dao: Rc<Dao<Array1<f32>>>, chunk_size: usize, num_neighbours: usize) -> (Vec<Vec<usize>>,Vec<Vec<f32>>) {
    let num_data = dao.num_data;
    let dims = dao.get_dim();
    let data = dao.get_data();
    let num_loops = num_data/chunk_size;

    let mut result_indices = vec![vec![0; num_neighbours]; num_data];
    let mut result_sims = vec![vec![f32::MAX; num_neighbours]; num_data];

    for i in 0..num_loops {

        let start_pos = i * chunk_size;
        let end_pos = num_data.min(start_pos + chunk_size);
        let chunk = &data.select(Axis(0),&(start_pos..end_pos).collect::<Vec<usize>>());

        let perm = RandomPermutation::new(num_data as u64).unwrap();
        let rand_ids = perm.iter().take(chunk_size).map( |x| x as usize ).collect::<Vec<usize>>(); // random data ids from whole data set

        let rand_data: Array1<Array1<f32>> = data.select(Axis(0), &rand_ids.as_slice() );           // select random vectors to compare
        let flat= rand_data.iter().flatten().cloned().collect::<Array1<f32>>();          // and make them into a Matrix
        let rand_data: Array2<f32> = flat.into_shape_with_order(((chunk_size,dims), Order::RowMajor)).unwrap();

        let chunk_transpose = &chunk.t();
        let flat= chunk_transpose.iter().flatten().cloned().collect::<Array1<f32>>();
        let chunk_transpose: Array2<f32> = flat.into_shape_with_order(((dims,chunk_size), Order::RowMajor)).unwrap();

        let chunk_dists: Array2<f32> = rand_data.dot( &chunk_transpose ); // matrix mult all the distances

        let (sorted_ords, sorted_dists) = arg_sort_array2(chunk_dists); // largest first, relative to to rand_data

        // get the num_neighbours closest original data indices

        let mut closest_dao_indices: Vec<Vec<usize>> = vec![vec![0; num_neighbours]; chunk_size];

        for row_index in 0..sorted_ords.len() {
            let row = &sorted_ords[row_index];
            for col in 0..num_neighbours {
                closest_dao_indices[row_index][col] = rand_ids[sorted_ords[row_index][col]] as usize;
            }
        }

        // Assign the nearest distances and neighbours to the result vectors
        for index in 0..sorted_dists.len()  {
            result_indices[start_pos + index] = closest_dao_indices[index].to_vec();
            result_sims[start_pos + index] = sorted_dists[index][0..num_neighbours].to_vec();
        }
    }
    (result_indices, result_sims)
}

pub fn getNNtable2(dao: Rc<Dao<Array1<f32>>>, neighbours: Vec<Vec<usize>>, similarities: Vec<Vec<f32>>, num_neighbours: usize,
                   k: usize, rho: f64, delta: f64 , reverse_list_size: usize ) {
    let num_data = dao.num_data;
    let data = &dao.get_data();

    let mut global_mins = similarities
        .iter()
        .map(|row| {
            row
                .iter()
                .map(|f| NonNan(*f))
                .min().unwrap().0
        })
        .collect::<Vec<f32>>();

    let mut iters = 0;
    let mut nn_new_flags: Vec<Vec<bool>> = vec![vec![true; num_neighbours]; num_data];
    let mut c = num_data as f64;

    while c > (num_data as f64) * delta {

        // phase 1

        let mut new: Vec<Vec<usize>> = vec![vec![0; num_neighbours]; num_data];
        let mut old: Vec<Vec<usize>> = vec![vec![0; num_neighbours]; num_data];

        // initialise old and new inline

        for row in 0..num_data {
            let flags = &nn_new_flags[row];
            let new_indices = flags.iter().filter(|x| **x == true).collect::<Vec<&bool>>();

            // random data ids from whole data set
            let perm = RandomPermutation::new(new_indices.len() as u64).unwrap(); // zero value returned?
            let sampled = perm.iter().take(((rho * new_indices.len() as f64)).round() as usize).map(|x| x as usize).collect::<Vec<usize>>(); // random data ids from whole data set

            (0..neighbours[row].len())
                .for_each(|i| {
                    if sampled.contains(&i) {
                        new[row][i] = neighbours[row][i];
                        nn_new_flags[row][i] = true;
                    } else {
                        old[row][i] = neighbours[row][i];
                        nn_new_flags[row][i] = false;
                    }
                });
        }

        // phase 2

        // initialise old' and new'

        let mut reverse: Vec<Vec<usize>> = vec![vec![0; reverse_list_size]; num_data];
        let mut reverse_sims = vec![vec![-1.0f32; reverse_list_size]; num_data];
        let mut reverse_ptr = vec![0; reverse_list_size];

        // calculate newPrime and oldPrime less stupidly...

        for row in 0..num_data {
            let all_ids = &neighbours[row];
            for id in 0..k {
                let this_id = &all_ids[id];
                let local_sim = similarities[row][id];
                let next_reverse_location = reverse_ptr[*this_id];

                if reverse_ptr[*this_id] <= k {
                    reverse_ptr[*this_id] = next_reverse_location;
                    reverse[*this_id][next_reverse_location] = row;
                    reverse_sims[*this_id][next_reverse_location] = local_sim;
                } else {
                    if let Some((position, value)) = reverse_sims[*this_id]
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.total_cmp(b))
                        .map(|(index, value)| (index, value)) {
                        if value < &local_sim {
                            reverse[*this_id][position] = row;
                            reverse_sims[*this_id][position] = local_sim;
                        }
                    } else {
                        panic!("Found None in reverse_sims");
                    }
                }
            }
        }


        // phase 3

        let c = 0;

        for row in 0..num_data {

            // was matlab nonzeros(A): returns a full column vector of the nonzero elements in A. The elements in v are ordered by columns.

            let old_row = old[row].iter().filter(|v| **v != 0).collect::<Vec<&usize>>();
            let new_row = new[row].iter().filter(|v| **v != 0).collect::<Vec<&usize>>();
            let mut prime_row = reverse[row].iter().filter(|v| **v != 0).collect::<Vec<&usize>>();

            if rho < 1.0 {
                // randomly shorten the prime_row vector
                prime_row = rand_perm(prime_row.len(), (rho * prime_row.len() as f64).round() as usize);
            }
            let new_row_union: Vec<usize> =
                if !new_row.is_empty() {
                    [&new_row[..], &prime_row[..]].concat()
                } else {
                    vec![]
                };

            // TODO these could well be wrong!!
            let old_data = old_row.iter().map(|i| dao.get_datum(**i)).collect::<Vec<Vec<f32>>>();
            let new_data = new_row.iter().map(|i| dao.get_datum(**i)).collect::<Vec<Vec<f32>>>();
            let new_union_data = new_row_union.iter().map(|i| dao.get_datum(**i)).collect::<Vec<Vec<f32>>>();

            // need new_union_data as a Matrix...
            let new_union_data = vecvec_to_ndarray(new_union_data);

            let new_new_sims = new_union_data.dot(&new_union_data.t());

            // separate for loops for the two distance tables...
            // for each pair of elements in the newNew list, their original ids

            // THIS IS LINE 146 of the text that is in richard_build_nns.txt (in matlab folder) and also below..

        }
    }
}

/*
				for newInd1 = 1 : length(newRowUnion) - 1
					u1Id = newRowUnion(newInd1);

					for newInd2 = newInd1 + 1 : length(newRowUnion)
						u2Id = newRowUnion(newInd2);
						% then get their similarity from the matrix
						thisSim = newNewSims(newInd1,newInd2);
						% is the current similarity greater than the biggest distance
						% in the row for u1Id? if it's not, then do nothing

						if thisSim > globalMinsInB(u1Id)
							% if it is, then u2Id actually can't already be there
							if sum(B(u1Id,:) == u2Id) == 0
								[~,wh] = min(nnSims(u1Id,:));
								B(u1Id,wh) = u2Id;
								nnNewFlags(u1Id,wh) = true;
								nnSims(u1Id,wh) = thisSim;
								globalMinsInB(u1Id) = min(nnSims(u1Id,:));
								c = c + 1;
							end
						end

						if globalMinsInB(u2Id) < thisSim
							if sum(B(u2Id,:) == u1Id) == 0
								[~,wh] = min(nnSims(u2Id,:));
								B(u2Id,wh) = u1Id;
								nnNewFlags(u2Id,wh) = true;
								nnSims(u2Id,wh) = thisSim;
								globalMinsInB(u2Id) = min(nnSims(u2Id,:));
								c = c + 1;
							end
						end
					end
				end // start 146

				%now do the news vs the olds, no reverse links
				newOldSims = newData * oldData';
				%and do the same for each pair of elements in the newRow/oldRow

				for newInd1 = 1 : length(newRow)
					u1Id = newRow(newInd1);
					for newInd2 = 1 : length(oldRow)
						u2Id = oldRow(newInd2);
						% then get their distance from the matrix
						thisSim = newOldSims(newInd1,newInd2);
						% is the current distance greater than the biggest distance
						% in the row for u1Id? if it's not, then do nothing
						if thisSim > globalMinsInB(u1Id)
							% if it is, then u2Id actually can't already be there
							if sum(B(u1Id,:) == u2Id) == 0
								[~,wh] = min(nnSims(u1Id,:));
								B(u1Id,wh) = u2Id;
								nnSims(u1Id,wh) = thisSim;
								nnNewFlags(u1Id,wh) = true;
								globalMinsInB(u1Id) = min(nnSims(u1Id,:));
								c = c + 1;
							end
						end

						if thisSim > globalMinsInB(u2Id)
							if sum(B(u2Id,:) == u1Id) == 0
								[~,wh] = min(nnSims(u2Id,:));
								B(u2Id,wh) = u1Id;
								nnSims(u2Id,wh) = thisSim;
								nnNewFlags(u2Id,wh) = true;
								globalMinsInB(u2Id) = min(nnSims(u2Id,:));
								c = c + 1;
							end // start at 206
						end // end of 205
					end // 187 start
		    end // line 185
	    end
	    phaseTimes(3,iters) = toc(phase3timer);
    end
end

 */






