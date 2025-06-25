use crate::dao_manager::{DaoManager, DaoStore};
use crate::table_initialisation::initialise_table_bsp_randomly;
use bits::container::{BitsContainer, Simd256x2};
use bits::evp::{matrix_dot, similarity_as_f32};
use bits::EvpBits;
use dao::Dao;
use ndarray::{s, Array1, Array2, ArrayView, ArrayView1, ArrayViewMut1, Axis};
use r_descent::functions::{fill_false_atomic, fill_selected};
use r_descent::{check_apply_update, get_slice_using_selectors, RDescent};
use rayon::prelude::*;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;
use utils::address::{GlobalAddress, LocalAddress, TableAddress};
use utils::{min_index_and_value_neighbourlarities, rand_perm, Nality};

pub fn into_big_knn_r_descent<C: BitsContainer, const W: usize>(
    daos: Vec<Rc<Dao<EvpBits<C, W>>>>,
    num_neighbours: usize,
    reverse_list_size: usize,
    delta: f64,
) -> RDescent {
    let num_data = daos.iter().map(|dao| dao.num_data).sum();

    let neighbourlarities = initialise_table_bsp_randomly(num_data, num_neighbours);

    make_big_knn_table2_bsp(
        daos,
        num_data,
        &neighbourlarities,
        num_neighbours,
        delta,
        reverse_list_size,
    );

    let ords = neighbourlarities.mapv(|x| x.id() as usize);
    let dists = neighbourlarities.mapv(|x| x.sim() as f32);

    RDescent {
        neighbours: ords,
        similarities: dists,
    }
}

pub fn make_big_knn_table2_bsp<C: BitsContainer, const W: usize>(
    daos: Vec<Rc<Dao<EvpBits<C, W>>>>,
    num_data: usize,
    neighbourlarities: &Array2<Nality>,
    num_neighbours: usize,
    delta: f64,
    reverse_list_size: usize,
) {
    log::warn!("Neighbourities length: {:?}", neighbourlarities.shape(),);

    let start_time = Instant::now();

    let dao_manager = DaoStore::new(daos);

    let mut iterations = 0;

    let mut neighbour_is_new =
        Array2::from_shape_fn((num_data, num_neighbours), |_| AtomicBool::new(true));

    let mut work_done: AtomicUsize = AtomicUsize::new(num_data); // a count of the number of times a similarity minimum of row has changed - measure of flux

    while work_done.load(std::sync::atomic::Ordering::SeqCst) > ((num_data as f64) * delta) as usize
    {
        // Matlab line 61
        // condition is fraction of lines whose min similarity has changed when this gets low - no much work done then stop.
        iterations += 1;

        log::debug!(
            "iterating: c: {} num_data: {} iters: {}",
            work_done.load(std::sync::atomic::Ordering::SeqCst),
            num_data,
            iterations
        );

        // phase 1

        let now = Instant::now();

        let mut new: Array2<Nality> =
            Array2::from_elem((num_data, num_neighbours), Nality::new_empty()); // Matlab line 65
        let mut old: Array2<Nality> =
            Array2::from_elem((num_data, num_neighbours), Nality::new_empty());

        // initialise old and new inline

        for row in 0..num_data {
            // in Matlab line 74
            let row_flags = neighbour_is_new.row_mut(row); // Matlab line 74

            // new_indices are the indices in this row whose flag is set to true (columns)

            let new_indices = row_flags // Matlab line 76
                .iter()
                .enumerate()
                .filter_map(|(index, flag)| {
                    if flag.load(Ordering::Relaxed) {
                        Some(index)
                    } else {
                        None
                    }
                })
                .collect::<Array1<usize>>();

            // old_indices are the indices in this row whose flag is set to false (intially there are none of these).

            let old_indices = row_flags // Matlab line 77
                .iter()
                .enumerate()
                .filter_map(|(index, flag)| {
                    if !flag.load(Ordering::Relaxed) {
                        Some(index)
                    } else {
                        None
                    }
                })
                .collect::<Array1<usize>>();

            // random data ids from whole data set
            // in matlab p = randperm(n,k) returns a row vector containing k unique integers selected randomly from 1 to n

            let sampled = rand_perm(
                new_indices.len(),
                (new_indices.len() as f64).round() as u64 as usize,
            );

            // sampled are random indices from new_indices

            let mut new_row_view: ArrayViewMut1<Nality> = new.row_mut(row);
            let mut old_row_view: ArrayViewMut1<Nality> = old.row_mut(row);
            let mut neighbour_row_view: ArrayViewMut1<AtomicBool> = neighbour_is_new.row_mut(row);

            fill_selected(
                &mut new_row_view,
                &neighbourlarities.row(row),
                &sampled.view(),
            ); // Matlab line 79

            fill_selected(
                &mut old_row_view,
                &neighbourlarities.row(row),
                &old_indices.view(),
            );

            fill_false_atomic(&mut neighbour_row_view, &sampled.view())
        }

        let after = Instant::now();
        log::debug!("Phase 1: {} ms", ((after - now).as_millis() as f64));

        // phase 2  Matlab line 88

        let now = Instant::now();

        // initialise old' and new'  Matlab line 90

        // // the reverse NN table  Matlab line 91
        // let mut reverse: Array2<usize> = Array2::from_elem((num_data, reverse_list_size), 0);
        // // all the distances from reverse NN table.
        // let mut reverse_sims: Array2<f32> =
        //     Array2::from_elem((num_data, reverse_list_size), -1.0f32);

        let mut reverse: Array2<Nality> =
            Array2::from_elem((num_data, reverse_list_size), Nality::new_empty());
        // reverse_ptr - how many reverse pointers for each entry in the dataset
        // FERDIA: brings us to 18GB here
        let mut reverse_count = Array1::from_elem(num_data, 0);

        // loop over all current entries in neighbours; add that entry to each row in the
        // reverse list if that id is in the forward NNs
        // there is a limit to the number of reverse ids we will store, as these
        // are in a zipf distribution, so we will add the most similar only

        for row in (0..num_data).map(|x| LocalAddress::new(x as u32)) {
            // Matlab line 97
            // all_ids are the forward links in the current id's row

            let this_row_neighbourlarities = &neighbourlarities.row(TableAddress::as_usize(&row)); // Matlab line 98
                                                                                                   // so for each one of these (there are k...):
            for id in 0..num_neighbours {
                // Matlab line 99 (updated)
                // get the id
                let this_id = this_row_neighbourlarities[id].id() as usize;
                // and how similar it is to the current id
                let local_sim = this_row_neighbourlarities[id].sim();

                // newForwardLinks = new(thisId,:);
                let new_forward_links = new.row(this_id);

                // forwardLinksDontContainThis = sum(newForwardLinks == i_phase2) == 0;
                let forward_links_dont_contain_this = !new_forward_links.iter().any(|x| {
                    dao_manager.table_addr_from_global_addr(&GlobalAddress::into(x.id())) == row
                });

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

                    if reverse_count[this_id] < reverse_list_size {
                        // if the list is not full
                        // update the reverse pointer list and the similarities

                        reverse[[this_id, reverse_count[this_id]]] =
                            Nality::new(local_sim, dao_manager.global_addr_from_table_addr(&row));
                        reverse_count[this_id] = reverse_count[this_id] + 1; // increment the count
                    } else {
                        // the list is full - so no need to do anything with counts
                        // but it is, so we will only add it if it's more similar than another one already there

                        let (position, value) =
                            min_index_and_value_neighbourlarities(&reverse.row(this_id)); // Matlab line 109
                        let value = value.sim();

                        if value < local_sim {
                            // Matlab line 110  if the value in reverse_sims is less similar we over write
                            reverse[[this_id, position as usize]] = Nality::new(
                                local_sim,
                                dao_manager.global_addr_from_table_addr(&row),
                            );
                        }
                    }
                }
            }
        }

        let after = Instant::now();
        log::debug!("Phase 2: {} ms", ((after - now).as_millis() as f64));

        // phase 3

        let now = Instant::now();

        work_done = AtomicUsize::new(0);

        // let updates = Updates::new(num_data); // TODO delete

        // let mut mutexes = vec![];
        // for i in 0..num_data {
        //     mutexes.push(Mutex::new(()));
        // }

        old.axis_iter_mut(Axis(0)) // Get mutable rows (disjoint slices)
            .enumerate()
            .zip(new.axis_iter_mut(Axis(0))
            )
            .par_bridge()
            .map(|((row, old_row), new_row)| {
                let binding = reverse
                    .row(row);


                let new_row_union: Array1<usize> = if new_row.len() == 0 {
                    // Matlab line 130
                    Array1::from(vec![])
                } else {
                    new_row
                        .iter()
                        .filter_map(|x| { if x.is_empty() { None } else { Some(x.id() as usize) } }) //<<<<<<<<< only take real values
                        .chain(binding
                            .iter()
                            .filter(|&x| !x.is_empty())
                            .map(|x| x.id() as usize))
                        .collect::<Array1<usize>>()
                };

                // index the data using the rows indicated in old_row
                let old_data = get_slice_using_multi_dao_selectors(
                    &dao_manager,
                    &old_row
                        .iter()
                        .map(|x| { x.id() as usize })
                        .filter(|global_address: &usize| dao_manager.is_mapped(GlobalAddress::into((*global_address).try_into().unwrap_or_else(|_| panic!("Cannot convert to u32"))))) // only look at addresses that are mapped.
                        .collect::<Array1<_>>().view() ); // Matlab line 136

                let new_data = get_slice_using_multi_dao_selectors(
                    &dao_manager,
                    &new_row
                        .iter()
                        .map(|x| x.id() as usize)
                        .filter(|global_address: &usize| dao_manager.is_mapped(GlobalAddress::into((*global_address).try_into().unwrap_or_else(|_| panic!("Cannot convert to u32"))))) // only look at addresses that are mapped.
                        .collect::<Array1<_>>().view() ); // Matlab line 137

                let new_union_data =
                    get_slice_using_multi_dao_selectors(&dao_manager, &new_row_union.view()); // Matlab line 137

                let new_new_sims =
                    matrix_dot(new_union_data.view(), new_union_data.view(), |a, b| {
                        similarity_as_f32(a, b)
                    });

                (
                    new_row,
                    old_row,
                    new_row_union,
                    new_new_sims,
                    new_data,
                    old_data,
                )
            })
            .for_each(
                |(new_row,
                     old_row,
                     new_row_union,
                     new_new_sims,
                     new_data,
                     old_data)| {
                    // Two for loops for the two distance tables (similarities and new_old_sims) for each pair of elements in the newNew list, their original ids
                    // First iterate over new_new_sims.. upper triangular (since distance table)

                    if new_row_union.len() >= 2 {
                        // must be at least 2 elements in the array because we are doing pair-wise comparisons.

                        for new_ind1 in 0..new_row_union.len() - 1 {
                            // Matlab line 144 (-1 since don't want the diagonal)
                            let u1_id = *new_row_union.get(new_ind1).unwrap_or_else(|| panic!("Illegal index of new_row_union at {new_ind1} length is: {}", new_row_union.len()));

                            for new_ind2 in new_ind1 + 1..new_row_union.len() {
                                // Matlab line 147
                                let u2_id = *new_row_union.get(new_ind2).unwrap_or_else(|| panic!("Illegal index of new_row_union at {new_ind2} length is: {}", new_row_union.len()));
                                // then get their similarity from the matrix
                                let this_sim = *new_new_sims.get((new_ind1, new_ind2)).unwrap_or_else(|| panic!("Illegal index of new_new_sims at {new_ind1},{new_ind2} Shape is: {:?}", new_new_sims.shape()));
                                // is the current similarity greater than the biggest distance
                                // in the row for u1_id? if it's not, then do nothing

                                check_apply_update(
                                    u1_id,
                                    u2_id as u32, // &Nality::new(this_sim, ),
                                    this_sim,
                                    &neighbour_is_new,
                                    neighbourlarities,
                                    &work_done,
                                );
                                check_apply_update(
                                    u2_id,
                                    u1_id as u32,  // &Nality::new(this_sim, ),
                                    this_sim,
                                    &neighbour_is_new,
                                    neighbourlarities,
                                    &work_done,
                                );
                            } // Matlab line 175
                        }

                        // now do the news vs the olds, no reverse links

                        let new_old_sims =
                            matrix_dot(new_data.view(),
                                       old_data.view(),
                                       |a, b| { similarity_as_f32(a, b) },
                            );

                        // and do the same for each pair of elements in the new_row/old_row

                        for new_ind1 in 0..new_row.len() {
                            // Matlab line 183  // rectangular matrix - need to look at all

                            let u1 = &new_row.get(new_ind1).unwrap_or_else(|| panic!("Illegal index of new_row at {new_ind1} length is: {}", new_row.len()));
                            for new_ind2 in 0..old_row.len() {
                                let u2 = &old_row.get(new_ind2).unwrap(); // Matlab line 186

                                // then get their distance from the matrix

                                let this_sim = *new_old_sims.get((new_ind1, new_ind2)).unwrap_or_else(|| panic!("Illegal index of new_old_sims at {new_ind1},{new_ind2} Shape is: {:?}", new_old_sims.shape()));

                                check_apply_update(
                                    u1.id() as usize,   // the new row
                                    u2.id(),            // <<<<<<<<<< the new index to be added to row
                                    this_sim,           // with this similarity
                                    &neighbour_is_new,
                                    neighbourlarities,
                                    &work_done,
                                );

                                check_apply_update(
                                    u2.id() as usize,
                                    u1.id(),
                                    this_sim,
                                    &neighbour_is_new,
                                    neighbourlarities,
                                    &work_done,
                                );
                            }
                        }
                    }
                },
            );

        let after = Instant::now();
        log::debug!("Phase 3: {} ms", ((after - now).as_millis() as f64));
    }

    let final_time = Instant::now();
    log::debug!(
        "Overall time 3: {} ms",
        ((final_time - start_time).as_millis() as f64)
    );
}

fn get_slice_using_multi_dao_selectors<C: BitsContainer, const W: usize>(
    source: &DaoStore<C, { W }>,
    selectors: &ArrayView1<usize>,
) -> Array1<EvpBits<C, W>> {
    todo!();
    // let mut sliced = Array1::uninit(selectors.len());
    //
    // for count in 0..selectors.len() {
    //
    //
    //     source
    //         .slice(s![selectors[count]])
    //         .assign_to(sliced.slice_mut(s![count]));
    // }
    //
    // unsafe { sliced.assume_init() }
}
