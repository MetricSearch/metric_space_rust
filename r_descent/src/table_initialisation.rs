//! Table initialisation code

use crate::functions::{
    get_slice_using_selected, insert_column_inplace, insert_index_at_position_1_inplace,
};
use crate::get_slice_using_selectors;
use bits::container::BitsContainer;
use bits::{evp::matrix_dot, evp::similarity_as_f32, EvpBits};
use dao::{Dao, DaoMatrix};
use ndarray::{s, Array2, Axis, Zip};
use rand::Rng;
use rayon::prelude::*;
use std::rc::Rc;
use std::time::Instant;
use utils::{arg_sort_big_to_small_2d, bytes_fmt, rand_perm, Nality};

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
        .into_par_iter()
        .zip(result_sims.axis_chunks_iter_mut(Axis(0), chunk_size))
        .enumerate()
        .for_each(|(i, (mut result_indices_chunk, mut result_sims_chunk))| {
            let real_chunk_size = result_sims_chunk.shape()[0];
            let start_pos = i * chunk_size;
            let end_pos = start_pos + real_chunk_size;

            let chunk = data.slice(s![start_pos..end_pos, 0..]);

            let original_row_ids = rand_perm(num_data, real_chunk_size); // random data ids from whole data set
            let rand_data = get_slice_using_selected(&data, &original_row_ids.view()); // a view of the original data points as a matrix
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

//
// pub fn get_nn_table2_m(
//     dao: Rc<DaoMatrix<f32>>,
//     mut neighbours: &mut Array2<usize>,
//     mut similarities: &mut Array2<f32>, // bigger is better
//     num_neighbours: usize,
//     delta: f64,
//     reverse_list_size: usize,
// ) {
//     let start_time = Instant::now();
//
//     let num_data = dao.num_data;
//     let dims = dao.get_dim();
//     let data = dao.get_data();
//
//     // Matlab lines refer to richard_build.txt file in the matlab dir
//
//     let mut iterations = 0;
//     let mut neighbour_is_new = Array2::from_elem((num_data, num_neighbours), true);
//     let mut work_done = num_data; // a count of the number of times a similarity minimum of row has changed - measure of flux
//
//     while work_done > ((num_data as f64) * delta) as usize {
//         // Matlab line 61
//         // condition is fraction of lines whose min similarity has changed when this gets low - no much work done then stop.
//         iterations += 1;
//
//         log::debug!(
//             "iterating: c: {} num_data: {} iters: {}",
//             work_done,
//             num_data,
//             iterations
//         );
//
//         // phase 1
//
//         let now = Instant::now();
//
//         let mut new: Array2<usize> = Array2::from_elem((num_data, num_neighbours), 0); // Matlab line 65
//         let mut old: Array2<usize> = Array2::from_elem((num_data, num_neighbours), 0);
//
//         // initialise old and new inline
//
//         for row in 0..num_data {
//             // in Matlab line 74
//             let row_flags = &neighbour_is_new.row_mut(row); // Matlab line 74
//
//             // new_indices are the indices in this row whose flag is set to true (columns)
//
//             let new_indices = row_flags // Matlab line 76
//                 .iter()
//                 .enumerate()
//                 .filter_map(|(index, flag)| if *flag { Some(index) } else { None })
//                 .collect::<Array1<usize>>();
//
//             // old_indices are the indices in this row whose flag is set to false (intially there are none of these).
//
//             let old_indices = row_flags // Matlab line 77
//                 .iter()
//                 .enumerate()
//                 .filter_map(|(index, flag)| if !*flag { Some(index) } else { None })
//                 .collect::<Array1<usize>>();
//
//             // random data ids from whole data set
//             // in matlab p = randperm(n,k) returns a row vector containing k unique integers selected randomly from 1 to n
//
//             let sampled = rand_perm(
//                 new_indices.len(),
//                 new_indices.len() as f64.round() as u64 as usize,
//             );
//
//             // sampled are random indices from new_indices
//
//             let mut new_row_view: ArrayViewMut1<usize> = new.row_mut(row);
//             let mut neighbour_row_view: ArrayViewMut1<bool> = neighbour_is_new.row_mut(row);
//
//             fill_selected(&mut new_row_view, &neighbours.row(row), &sampled.view()); // Matlab line 79
//             fill_selected(&mut new_row_view, &neighbours.row(row), &old_indices.view());
//             fill_false(&mut neighbour_row_view, &sampled.view())
//         }
//
//         let after = Instant::now();
//         log::debug!("Phase 1: {} ms", ((after - now).as_millis() as f64));
//
//         // phase 2  Matlab line 88
//
//         let (reverse, _reverse_sims) = get_new_reverse_links_not_in_forward(
//             &neighbours,
//             &similarities,
//             reverse_list_size,
//             &new,
//         );
//
//         // phase 3
//
//         let now = Instant::now();
//
//         work_done = 0;
//
//         let mut updates = Updates::new(num_data);
//
//         old.axis_iter_mut(Axis(0)) // Get mutable rows (disjoint slices)
//             .enumerate()
//             .zip(new.axis_iter_mut(Axis(0)))
//             .par_bridge()
//             .map(|((row, old_row), new_row)| {
//                 let mut reverse_link_row: Array1<usize> = reverse
//                     .row(row)
//                     .iter()
//                     .filter(|&&v| v != 0)
//                     .copied()
//                     .collect();
//
//                 let mut new_row_union: Array1<usize> = if new_row.len() == 0 {
//                     // Matlab line 130
//                     Array1::from(vec![])
//                 } else {
//                     new_row
//                         .iter()
//                         .copied()
//                         .chain(reverse_link_row.iter().copied())
//                         .collect::<Array1<usize>>() // <<<<< 2 row copies here
//                 };
//
//                 let new_row_union_len = new_row_union.len();
//
//                 // index the data using the rows indicated in old_row
//                 let old_data =
//                     get_slice_using_selected(&data, &old_row.view(), [old_row.len(), dims]); // Matlab line 136
//                 let new_data =
//                     get_slice_using_selected(&data, &new_row.view(), [new_row.len(), dims]); // Matlab line 137
//                 let new_union_data = get_slice_using_selected(
//                     &data,
//                     &new_row_union.view(),
//                     [new_row_union_len, dims],
//                 ); // Matlab line 137
//
//                 let new_new_sims: Array2<f32> = new_union_data.dot(&new_union_data.t()); // Matlab line 139
//
//                 (
//                     row,
//                     new_row,
//                     old_row,
//                     new_row_union,
//                     new_new_sims,
//                     new_data,
//                     old_data,
//                 )
//             })
//             .for_each(
//                 |(row, new_row, old_row, new_row_union, new_new_sims, new_data, old_data)| {
//                     // Two for loops for the two distance tables (similarities and new_old_sims) for each pair of elements in the newNew list, their original ids
//                     // First iterate over new_new_sims.. upper triangular (since distance table)
//
//                     for new_ind1 in 0..new_row_union.len() - 1 {
//                         // Matlab line 144 (-1 since don't want the diagonal)
//                         let u1_id = new_row_union[new_ind1];
//
//                         for new_ind2 in new_ind1 + 1..new_row_union.len() {
//                             // Matlab line 147
//                             let u2_id = new_row_union[new_ind2];
//                             // then get their similarity from the matrix
//                             let this_sim = new_new_sims[[new_ind1, new_ind2]];
//                             // is the current similarity greater than the biggest distance
//                             // in the row for u1_id? if it's not, then do nothing
//
//                             if this_sim > minimum_in(&similarities.row(u1_id)) {
//                                 // Matlab line 154 // global_mins[u1_id]
//                                 // if it is, then u2_id actually can't already be there
//
//                                 updates.add(u1_id, u2_id, this_sim);
//                             }
//
//                             if minimum_in(&similarities.row(u2_id)) < this_sim {
//                                 // Matlab line 166 // was global_mins[u2_id]
//                                 updates.add(u2_id, u1_id, this_sim);
//                             }
//                         } // Matlab line 175
//                     }
//
//                     // nnw do the news vs the olds, no reverse links
//                     // newOldSims = newData * oldData';
//
//                     let new_old_sims = new_data.dot(&old_data.t());
//
//                     // and do the same for each pair of elements in the new_row/old_row
//
//                     for new_ind1 in 0..new_row.len() {
//                         // Matlab line 183  // rectangular matrix - need to look at all
//                         let u1_id = new_row[new_ind1];
//                         for new_ind2 in 0..old_row.len() {
//                             let u2_id = old_row[new_ind2]; // Matlab line 186
//                                                            // then get their distance from the matrix
//                             let this_sim = new_old_sims[[new_ind1, new_ind2]];
//                             // is the current distance greater than the biggest distance
//                             // in the row for u1_id? if it's not, then do nothing
//                             if this_sim > minimum_in(&similarities.row(u1_id)) {
//                                 // Matlab line 191 // global_mins[u1_id]
//                                 // if it is, then u2Id actually can't already be there
//                                 updates.add(u1_id, u2_id, this_sim);
//                             }
//
//                             if this_sim > minimum_in(&similarities.row(u2_id)) {
//                                 // Matlab line 203 // was global_mins[u2_id]
//                                 updates.add(u2_id, u1_id, this_sim);
//                             }
//                         }
//                     }
//                 },
//             );
//
//         let after = Instant::now();
//         log::debug!("Phase 3: {} ms", ((after - now).as_millis() as f64));
//     }
//
//     let final_time = Instant::now();
//     log::debug!(
//         "Overall time 3: {} ms",
//         ((final_time - start_time).as_millis() as f64)
//     );
// }

//************** EvpBits impl below here **************

pub fn initialise_table_bsp_randomly(rows: usize, columns: usize) -> Array2<Nality> {
    log::info!("Randomly initializing table bsp, rows: {rows} neighbours: {columns}");
    let start_time = Instant::now();

    let mut rng = rand::rng();
    let nalities: Vec<Nality> = (0..rows * columns)
        .map(|_| {
            let rand_index = rng.random_range(0..rows); // pick random row index
            Nality::new_empty_index(rand_index as u32)
        })
        .collect();

    let mut nalities = Array2::from_shape_vec((rows, columns), nalities)
        .expect("Shape mismatch during initialisation");

    // overwrite first entry with a new nality of itself and 0
    for row in 0..nalities.nrows() {
        nalities[[row, 0]] = Nality::new(f32::MAX, row as u32);
    }

    let end_time = Instant::now();
    log::debug!(
        "Initialistion in {:?}ms",
        ((end_time - start_time).as_millis() as f64)
    );

    nalities
}

pub fn initialise_table_bsp<C: BitsContainer, const W: usize>(
    dao: Rc<Dao<EvpBits<C, W>>>,
    chunk_size: usize,
    num_neighbours: usize,
) -> Array2<Nality> {
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
        .into_par_iter()
        .zip(result_sims.axis_chunks_iter_mut(Axis(0), chunk_size))
        .enumerate()
        .for_each(|(i, (mut result_indices_chunk, mut result_sims_chunk))| {
            let real_chunk_size = result_sims_chunk.shape()[0];
            let start_pos = i * chunk_size;
            let end_pos = start_pos + real_chunk_size;

            let chunk = data.slice(s![start_pos..end_pos]);

            let original_row_ids = rand_perm(num_data, real_chunk_size); // random data ids from whole data set
            let rand_data = get_slice_using_selectors(&data, &original_row_ids.view()); // a view of the original data points as a matrix

            let chunk_dists = matrix_dot(chunk, rand_data.view(), |a, b| similarity_as_f32(a, b)); // matrix mult all the distances - all relative to the original_rows

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

    let indices = insert_index_at_position_1_inplace(result_indices);
    let sims = insert_column_inplace(result_sims, 1224.0);

    // Makes neighbourlarities from similarities and ids
    let xx = Zip::from(&indices)
        .and(&sims)
        .map_collect(|&id, &sim| Nality::new(sim, id as u32));

    // println!("First row initialisation: {:#?}", xx.row(0)); // TODO fix this

    xx
}
