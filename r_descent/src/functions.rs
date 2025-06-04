//! Utility functions for lib.rs

use dao::Dao;
use ndarray::{s, Array, Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, Ix1};
use std::ptr;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use utils::{min_index_and_value, min_index_and_value_neighbourlarities, Nality};

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
    let mut reverse: Array2<usize> =
        Array2::from_elem((num_data, reverse_list_size), u32::MAX as usize);
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
    let mut reverse: Array2<usize> =
        Array2::from_elem((num_data, reverse_list_size), u32::MAX as usize);
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

            // log::debug!(
            //     "Row {} col {} next_id {} sim {} neighbours of next {} don't contain {} row {}",
            //     row,
            //     col,
            //     next_id_in_row,
            //     next_sim_in_row,
            //     neighbours_of_next_id_in_row,
            //     row,
            //     neighbours_of_next_dont_contain_current_row
            // );

            // if the reverse list isn't full, we will just add this one
            // this adds to a priority queue and keeps track of max
            // We are trying to find a set of reverse near neighbours with the
            // biggest similarity of size reverse_list_size.
            // first find all the forward links containing the row

            if neighbours_of_next_dont_contain_current_row {
                //log::debug!("count is {} ", reverse_count[*next_id_in_row]);
                if reverse_count[*next_id_in_row] < reverse_list_size {
                    // if the list is not full
                    // update the reverse pointer list and the similarities
                    // log::debug!(
                    //     "Adding row {} refers to {} insert position {}",
                    //     row,
                    //     *next_id_in_row,
                    //     reverse_count[*next_id_in_row]
                    // );

                    reverse[[*next_id_in_row, reverse_count[*next_id_in_row]]] = row;
                    reverse_sims[[*next_id_in_row, reverse_count[*next_id_in_row]]] =
                        next_sim_in_row; // pop that in too
                    reverse_count[*next_id_in_row] = reverse_count[*next_id_in_row] + 1;
                // increment the count
                } else {
                    // it is full, so we will only add it if it's more similar than another one already there
                    let (position, value) = min_index_and_value(&reverse_sims.row(*next_id_in_row)); // Matlab line 109
                                                                                                     // log::debug!(
                                                                                                     //     "full min index in {} and value of row {} are {} {}",
                                                                                                     //     &reverse_sims.row(*next_id_in_row),
                                                                                                     //     row,
                                                                                                     //     position,
                                                                                                     //     value
                                                                                                     // );
                    if value < next_sim_in_row {
                        // Matlab line 110  if the value in reverse_sims is less similar we over write
                        //log::debug!("overwriting");
                        reverse[[*next_id_in_row, position]] = row; // replace the old min with the new sim value
                        reverse_sims[[*next_id_in_row, position]] = next_sim_in_row;
                    }
                }
            }
        }
    }

    (reverse, reverse_sims)
}

// Same as function above without new parameter.
pub fn get_reverse_nality_links_not_in_forward(
    neighbours: &Array2<Nality>,
    reverse_list_size: usize,
) -> Array2<Nality> {
    // initialise old' and new'  Matlab line 90
    // the reverse NN table  Matlab line 91
    let num_neighbours = neighbours.ncols();
    let num_data = neighbours.nrows();
    let mut reverse: Array2<Nality> = Array2::from_elem(
        (num_data, reverse_list_size),
        Nality::new(f32::MIN, u32::MAX),
    );

    // reverse_count - how many reverse pointers for each entry in the dataset
    let mut reverse_count = Array1::from_elem(num_data, 0);

    // loop over all current entries in neighbours; add that entry to each row in the
    // reverse list if that id is in the forward NNs
    // there is a limit to the number of reverse ids we will store, as these
    // are in a zipf distribution, so we will add the most similar only

    for row in 0..num_data {
        // Matlab line 97
        // all_ids are the forward links in the current id's row
        let current_row = &neighbours.row(row); // Matlab line 98
                                                // so for each one of these (there are k...):
        for col in 0..num_neighbours {
            // Matlab line 99 (updated)
            // get the id
            let next_id_in_row = current_row[col].id();
            // and how similar it is to the current id
            let next_sim_in_row = current_row[col].sim();

            let neighbours_of_next_id_in_row = neighbours.row(next_id_in_row as usize);

            let neighbours_of_next_dont_contain_current_row = !neighbours_of_next_id_in_row
                .iter()
                .any(|x| x.id() == row as u32);

            // log::debug!(
            //     "Row {} col {} next_id {} sim {} neighbours of next {}",
            //     row,
            //     col,
            //     next_id_in_row,
            //     next_sim_in_row,
            //     neighbours_of_next_dont_contain_current_row
            // );

            // if the reverse list isn't full, we will just add this one
            // this adds to a priority queue and keeps track of max
            // We are trying to find a set of reverse near neighbours with the
            // biggest similarity of size reverse_list_size.
            // first find all the forward links containing the row

            if neighbours_of_next_dont_contain_current_row {
                //log::debug!("count is {} ", reverse_count[next_id_in_row as usize]);
                if reverse_count[next_id_in_row as usize] < reverse_list_size {
                    // if the list is not full
                    // update the reverse pointer list and the similarities
                    // log::debug!(
                    //     "Adding row {} refers to {} insert position {}",
                    //     row,
                    //     next_id_in_row,
                    //     reverse_count[next_id_in_row as usize]
                    // );

                    reverse[[
                        next_id_in_row as usize,
                        reverse_count[next_id_in_row as usize],
                    ]] = Nality::new(next_sim_in_row, row as u32);
                    reverse_count[next_id_in_row as usize] =
                        reverse_count[next_id_in_row as usize] + 1;
                // increment the count
                } else {
                    // it is full, so we will only add it if it's more similar than another one already there
                    let (index, nality) = min_index_and_value_neighbourlarities(
                        &reverse.row(next_id_in_row as usize),
                    ); // Matlab line 109

                    if nality.sim() < next_sim_in_row {
                        // Matlab line 110  if the value in reverse_sims is less similar we over write
                        // debug!("overwriting");
                        reverse[[next_id_in_row as usize, index]] =
                            Nality::new(next_sim_in_row, row as u32); // replace the old min with the new sim value
                    }
                }
            }
        }
    }

    reverse
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

        // log::debug!(
        //     "Reverse links: {:?} reverse sims: {:?}",
        //     reverse_links.0,
        //     reverse_links.1
        // );

        assert_eq!(reverse_links.0, gt_links);
    }
}

pub fn get_selected_data(
    dao: Rc<Dao<Array1<f32>>>,
    dims: usize,
    old_row: &Vec<usize>,
) -> Array2<f32> {
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

pub fn fill_false(row: &mut ArrayViewMut1<bool>, selector: &ArrayView1<usize>) {
    for i in 0..selector.len() {
        row[selector[i]] = false;
    }
}

pub fn fill_false_atomic(row: &mut ArrayViewMut1<AtomicBool>, selector: &ArrayView1<usize>) {
    for i in 0..selector.len() {
        row[selector[i]].store(false, Ordering::Relaxed);
    }
}

pub fn fill_selected(
    to_fill: &mut ArrayViewMut1<Nality>,
    fill_from: &ArrayView1<Nality>,
    selector: &ArrayView1<usize>,
) {
    for (i, &sel_index) in selector.iter().enumerate() {
        to_fill[i] = fill_from[sel_index].clone();
    }
}

pub fn get_slice_using_selected(
    source: &ArrayView2<f32>,
    selectors: &ArrayView1<usize>,
) -> Array2<f32> {
    let mut sliced = Array2::uninit([selectors.len(), source.ncols()]);

    for count in 0..selectors.len() {
        // was result_shape
        source
            .slice(s![selectors[count], 0..])
            .assign_to(sliced.slice_mut(s![count, 0..]));
    }

    unsafe { sliced.assume_init() }
}

pub fn get_1_d_slice_using_selected<T: Clone>(
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

pub fn get_selectors_from_flags(selectors: &Array1<bool>) -> Array1<usize> {
    let vec = selectors
        .into_iter()
        .enumerate()
        .filter_map(|(index, value)| if *value { Some(index) } else { None })
        .collect();

    Array1::from_vec(vec)
}

/// Selects and copies specific elements from a 1D array view into a new 1D array.
///
/// This function creates a new 1D array by selecting elements from the input `source`
/// using the indices provided in `selectors`. The resulting array will have the same
/// length as `selectors`, and elements will appear in the specified order.
///
/// # Type Parameters
///
/// - `T`: The element type, which must implement `Clone`.
///
/// # Parameters
///
/// - `source`: A 1D array view (`ArrayView1<T>`) from which elements will be selected.
/// - `selectors`: A 1D array view (`ArrayView1<usize>`) containing the indices of the
///   elements to extract from `source`. Each index must be within bounds of `source`.
///
/// # Returns
///
/// A new `Array1<T>` containing the selected elements in the order given by `selectors`.
///
/// # Panics
///
/// Panics if any index in `selectors` is out of bounds for `source`.
///
/// # Example
///
/// ```rust
/// use ndarray::{array, ArrayView1};
///
/// let source = array![10, 20, 30, 40];
/// let selectors = array![2, 0, 3];
/// let result = get_slice_using_selectors(&source.view(), &selectors.view());
/// assert_eq!(result, array![30, 10, 40]);
/// ```
///
/// # Safety
///
/// Internally, this function uses uninitialized memory to improve performance.
/// It ensures all elements are properly written before being returned, so it's
/// memory-safe when used correctly.
pub fn get_slice_using_selectors<T: Clone>(
    source: &ArrayView1<T>,
    selectors: &ArrayView1<usize>,
) -> Array1<T> {
    let mut sliced = Array1::uninit(selectors.len());

    for count in 0..selectors.len() {
        // was result_shape
        source
            .slice(s![selectors[count]])
            .assign_to(sliced.slice_mut(s![count]));
    }

    unsafe { sliced.assume_init() }
}

/// Selects and copies specific rows from a 2D array view into a new 2D array.
///
/// This function creates a new 2D array by selecting rows from the input `source` array
/// using the indices provided in `selectors`. The output array will have the same number
/// of columns as `source`, and the number of rows will match the length of `selectors`.
///
/// # Type Parameters
///
/// - `T`: The element type, which must implement `Clone`.
///
/// # Parameters
///
/// - `source`: A 2D array view (`ArrayView2<T>`) from which rows will be selected.
/// - `selectors`: A 1D array view (`ArrayView1<usize>`) containing the indices of the rows
///   to extract from `source`. Each index must be within bounds of `source`'s row count.
///
/// # Returns
///
/// A new `Array2<T>` containing the selected rows in the order given by `selectors`.
///
/// # Panics
///
/// Panics if any index in `selectors` is out of bounds for `source`.
///
/// # Example
///
/// ```rust
/// use ndarray::{array, ArrayView2, ArrayView1};
///
/// let source = array![[1, 2], [3, 4], [5, 6]];
/// let selectors = array![2, 0];
/// let result = get_2_d_slice_using(&source.view(), &selectors.view());
/// assert_eq!(result, array![[5, 6], [1, 2]]);
/// ```
///
/// # Safety
///
/// This function uses uninitialized memory internally for performance, but ensures all elements
/// are properly written before returning. Therefore, it is memory-safe under correct use.
pub fn get_2_d_slice_using<T: Clone>(
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

/// Inserts a new column at the beginning (index 0) of a 2D array in-place.
///
/// This function takes ownership of a 2D `Array2<f32>` with one extra column of capacity
/// and inserts a new column of values at the first position (column 0), shifting all existing
/// columns to the right. The inserted column is filled with the given `new_col_val`.
///
/// # Parameters
///
/// - `array`: A 2D array (`Array2<f32>`) with dimensions `(nrows, ncols + 1)`, where the extra
///   column provides the necessary capacity to shift elements in-place.
/// - `new_col_val`: The value to insert into the new first column for every row.
///
/// # Returns
///
/// A new `Array2<f32>` with the inserted column at the start, having the same shape as the input.
///
/// # Safety
///
/// This function uses unsafe code to manipulate the raw memory of the array for performance
/// reasons. It assumes the array has one additional column of capacity beyond the actual data
/// (`ncols + 1`), and that the memory layout is contiguous in row-major order.
///
/// # Panics
///
/// Panics if the input array has zero columns (i.e., `ncols_plus_1 == 0`).
///
/// # Example
///
/// ```rust
/// use ndarray::array;
/// let mut input = ndarray::Array2::<f32>::zeros((3, 4)); // 3 rows, 4 columns (with capacity)
/// let result = insert_column_inplace(input, 1.0);
/// assert_eq!(result.column(0), ndarray::Array1::from_elem(3, 1.0));
/// ```
pub fn insert_column_inplace(mut array: Array2<f32>, new_col_val: f32) -> Array2<f32> {
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
pub fn insert_index_at_position_1_inplace(mut array: Array2<usize>) -> Array2<usize> {
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
