use crate::dao_manager::{get_ranges, DaoManager, DaoStore};
use crate::NalityNNTable;
use bits::container::{BitsContainer, Simd256x2};
use bits::evp::{matrix_dot, similarity_as_f32};
use bits::EvpBits;
use dao::Dao;
use log::debug;
use ndarray::{
    concatenate, s, Array1, Array2, ArrayBase, ArrayView, ArrayView1, ArrayView2, ArrayViewMut1,
    ArrayViewMut2, Axis, CowArray, Ix2, OwnedRepr,
};
use r_descent::functions::{fill_false_atomic_from_selectors, fill_selected};
use r_descent::initialise_table_bsp_randomly_overwrite_row_0;
use r_descent::{check_apply_update, get_slice_using_selectors, RDescent};
use rayon::prelude::*;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;
use utils::address::{GlobalAddress, LocalAddress};
use utils::{min_index_and_value_neighbourlarities, minimum_in_nality, rand_perm, Nality};

pub fn into_big_knn_r_descent<C: BitsContainer, const W: usize>(
    daos: Vec<Dao<EvpBits<C, W>>>,
    num_neighbours: usize,
    reverse_list_size: usize,
    delta: f64,
    start_index: u32,
) -> NalityNNTable {
    let num_data = daos
        .iter()
        .map(|dao| {
            println!("xxxxx {}", dao.num_data);
            dao.num_data
        })
        .sum();

    println!("Num Data: {}", num_data);

    let neighbourlarities =
        initialise_table_bsp_randomly_overwrite_row_0(num_data, num_neighbours, start_index);

    let dao_manager = DaoStore::new(daos);

    // check_neighbours(&neighbourlarities, &dao_manager); // Debug check - not needed.

    make_big_knn_table2_bsp(
        dao_manager,
        num_data,
        &neighbourlarities,
        num_neighbours,
        delta,
        reverse_list_size,
    );

    NalityNNTable {
        nalities: neighbourlarities,
    }
}

pub fn make_big_knn_table2_bsp<C: BitsContainer, const W: usize>(
    dao_manager: DaoStore<C, W>,
    num_data: usize,
    neighbourlarities: &Array2<Nality>,
    num_neighbours: usize,
    delta: f64,
    reverse_list_size: usize,
) {
    log::debug!(
        "Making NN table with shape: {:?}",
        neighbourlarities.shape(),
    );

    let start_time = Instant::now();

    let mut iterations = 0;

    let mut neighbour_is_new: Array2<AtomicBool> =
        Array2::from_shape_fn((num_data, num_neighbours), |_| AtomicBool::new(true));

    let mut work_done: AtomicUsize = AtomicUsize::new(num_data); // a count of the number of times a similarity minimum of row has changed - measure of flux

    while work_done.load(Ordering::SeqCst) > ((num_data as f64) * delta) as usize {
        // Matlab line 61
        // condition is fraction of lines whose min similarity has changed when this gets low - no much work done then stop.

        // Phase 1 (was phase 2)  Matlab line 88

        let now = Instant::now();

        // Take a copy of the state of the world
        let mut old_neighbour_state = &mut neighbourlarities.clone();

        let old_flags: Array2<AtomicBool> = // clone the atomic flags array
            Array2::from_shape_fn(neighbour_is_new.dim(), |(i, j)| {
                AtomicBool::new(neighbour_is_new[(i, j)].load(Ordering::Relaxed))
            });

        let (reverse_links, reverse_count) = create_reverse_links(
            &dao_manager,
            num_data,
            &neighbourlarities.view(),
            num_neighbours,
            reverse_list_size,
            &mut old_neighbour_state.view(),
            &old_flags.view(),
        );

        let after = Instant::now();
        log::debug!("Phase 1: {} ms", ((after - now).as_millis() as f64));

        // Phase 2 (was phase 3)

        let now = Instant::now();

        work_done = AtomicUsize::new(0);

        old_neighbour_state
            .axis_iter(Axis(0)) // iterate over the rows
            .enumerate()
            .par_bridge()
            .map(|(row_index, nalities)| {
                let old_row_state = old_neighbour_state.row(row_index);

                let newly_updated_nailities_in_row: Vec<_> = old_row_state // may contain unmapped data
                    .into_iter()
                    .enumerate()
                    .filter(|(column, nality)| old_flags[[row_index, *column]].load(Ordering::Relaxed))
                    .map(|x| x.1)
                    .collect();

                let not_updated_nalities_in_row: Vec<_> = old_row_state // may contain unmapped data
                    .into_iter()
                    .enumerate()
                    .filter(|(column, nality)| !old_flags[[row_index, *column]].load(Ordering::Relaxed))
                    .map(|x| x.1)
                    .collect();

                let reverse_row_links = reverse_links.row(row_index); // may contain unmapped data

                let new_mapped_updated_forward_and_reverse_links: Array1<GlobalAddress> = if newly_updated_nailities_in_row.len() == 0 { // may contain unmapped data
                    // Matlab line 130
                    Array1::from(vec![])
                } else {
                    newly_updated_nailities_in_row
                        .iter()
                        .filter_map(|x| { if ! dao_manager.is_mapped(&x.id()) || x.is_empty() { None } else { Some(x.id()) } })
                        .chain(reverse_row_links
                            .iter()
                            .filter(|&x| dao_manager.is_mapped(&x.id()) || !x.is_empty())
                            .map(|x| x.id()))
                        .collect::<Array1<GlobalAddress>>()
                };

                // index the data using the rows indicated in old_row
                let not_updated_mapped_row_data = get_slice_using_multi_dao_selectors( // an array of evps selected from the old row for mapped entities
                                                                                       &dao_manager,
                                                                                       &not_updated_nalities_in_row
                                                                                           .iter()
                                                                                           .map(|x| { x.id() })
                                                                                           .filter(|global_address: &GlobalAddress| dao_manager.is_mapped(global_address)) // only look at addresses that are mapped
                                                                                           .collect::<Array1<GlobalAddress>>().view()); // Matlab line 136

                let new_mapped_row_data = //can only slice mapped data
                    get_slice_using_multi_dao_selectors(
                        &dao_manager,
                        &newly_updated_nailities_in_row // may contain data that is mapped and unmapped
                            .iter()
                            .map(|x| x.id())
                            .filter(|global_address: &GlobalAddress| dao_manager.is_mapped(global_address)) // only look at addresses that are mapped
                            .collect::<Array1<GlobalAddress>>().view()); // Matlab line 137

                let new_mapped_forward_and_reverse_data = //can only slice mapped data
                    get_slice_using_multi_dao_selectors(&dao_manager,
                                                        &new_mapped_updated_forward_and_reverse_links // may contain data that is mapped and unmapped
                                                            .iter()
                                                            .map(|x| *x)
                                                            .filter(|global_address: &GlobalAddress| dao_manager.is_mapped(&global_address)) // only look at addresses that are mapped //<< TODO DELETE ME
                                                            .collect::<Array1<GlobalAddress>>().view()); // Matlab line 137

                let new_mapped_forward_and_reverse_sims =
                    matrix_dot(new_mapped_forward_and_reverse_data.view(), new_mapped_forward_and_reverse_data.view(), |a, b| {
                        similarity_as_f32(a, b)
                    });

                (
                    newly_updated_nailities_in_row,
                    not_updated_nalities_in_row,
                    new_mapped_updated_forward_and_reverse_links,
                    new_mapped_forward_and_reverse_sims,
                    new_mapped_row_data,
                    not_updated_mapped_row_data,
                    reverse_row_links,
                )
            })
            .for_each(
                |(newly_updated_nailities_in_row,
                     not_updated_nalities_in_row,
                     newly_updated_forward_and_reverse_links,
                     new_mapped_forward_and_reverse_sims,
                     new_mapped_row_data,
                     old_mapped_row_data,
                     reverse_row_links
                 )| {
                    // Two for loops for the two distance tables (similarities and new_old_sims) for each pair of elements in the newNew list, their original ids
                    // First iterate over new_new_sims.. upper triangular (since distance table)

                    if newly_updated_forward_and_reverse_links.len() >= 2 {
                        // must be at least 2 elements in the array because we are doing pair-wise comparisons.

                        for new_ind1 in 0..newly_updated_forward_and_reverse_links.len() - 1 {
                            // Matlab line 144 (-1 since don't want the diagonal)
                            let u1_id = *newly_updated_forward_and_reverse_links.get(new_ind1).unwrap_or_else(|| panic!("Illegal index of newly_updated_forward_and_reverse_links at {new_ind1} length is: {}", newly_updated_forward_and_reverse_links.len()));

                            for new_ind2 in new_ind1 + 1..newly_updated_forward_and_reverse_links.len() {
                                // Matlab line 147
                                let u2_id = *newly_updated_forward_and_reverse_links.get(new_ind2).unwrap_or_else(|| panic!("Illegal index of newly_updated_forward_and_reverse_links at {new_ind2} length is: {}", newly_updated_forward_and_reverse_links.len()));
                                // then get their similarity from the matrix
                                let this_sim = *new_mapped_forward_and_reverse_sims.get((new_ind1, new_ind2)).unwrap_or_else(|| panic!("Illegal index of new_mapped_forward_and_reverse_sims at {new_ind1},{new_ind2} Shape is: {:?}", new_mapped_forward_and_reverse_sims.shape()));
                                // is the current similarity greater than the biggest distance
                                // in the row for u1_id? if it's not, then do nothing

                                // if (u2_id.as_usize() == 200001) {
                                //     println!("new_ind2 {new_ind2} U1 ID: global: {:?} local: {} | U2 ID global: {:?} local: {}  | Sim: {} | Current Row {:?}",
                                //              u1_id,
                                //              u1_id.as_usize(),
                                //              u2_id,
                                //              u2_id.as_usize(),
                                //              this_sim,
                                //              neighbourlarities.row(1));
                                //     println!("newly_updated_forward_and_reverse_links {:?}", newly_updated_forward_and_reverse_links);
                                //     println!("reverse_row_links {:?},", reverse_row_links);
                                // }

                                check_apply_update_wrapper(
                                    u1_id,
                                    u2_id,
                                    this_sim,
                                    &neighbour_is_new,
                                    &neighbourlarities,
                                    &work_done,
                                    &dao_manager,
                                );

                                check_apply_update_wrapper(
                                    u2_id,
                                    u1_id,
                                    this_sim,
                                    &neighbour_is_new,
                                    &neighbourlarities,
                                    &work_done,
                                    &dao_manager,
                                );
                            } // Matlab line 175
                        }

                        // now do the news vs the olds, no reverse links

                        let new_old_sims = if old_mapped_row_data.len() == 0 {
                            Array2::zeros((0, 0))
                        } else {
                            matrix_dot(new_mapped_row_data.view(),
                                       old_mapped_row_data.view(),
                                       |a, b| { similarity_as_f32(a, b) })
                        };

                        // Ben's suggestion - Re-insert unmapped nalities into new_old_sims

                        // and do the same for each pair of elements in the new_row/old_row

                        for new_row_index_1 in 0..newly_updated_nailities_in_row.len() {
                            // Matlab line 183  // rectangular matrix - need to look at all

                            let u1 = &newly_updated_nailities_in_row.get(new_row_index_1).unwrap_or_else(|| panic!("Illegal index of new_row at {new_row_index_1} length is: {}", newly_updated_nailities_in_row.len()));
                            for new_row_index_2 in 0..not_updated_nalities_in_row.len() {
                                let u2 = &not_updated_nalities_in_row.get(new_row_index_2).unwrap(); // Matlab line 186

                                // then get their distance from the matrix
                                let this_sim = *new_old_sims.get((new_row_index_1, new_row_index_2))
                                    .unwrap_or_else(|| panic!("Illegal index of new_old_sims at {new_row_index_1},{new_row_index_2} Shape is: {:?}", new_old_sims.shape()));

                                // if (u2.id().as_usize() == 200001) {
                                //     println!("new_row_index_2 {new_row_index_2} U1 ID: global: {:?} local: {} | U2 ID global: {:?} local: {}  | Sim: {} | Current Row {:?}",
                                //              u1.id(),
                                //              u1.id().as_usize(),
                                //              u2.id(),
                                //              u2.id().as_usize(),
                                //              this_sim,
                                //              neighbourlarities.row(1));
                                //     println!("newly_updated_forward_and_reverse_links {:?}", newly_updated_forward_and_reverse_links);
                                //     println!("reverse_row_links {:?},", reverse_row_links);
                                // }

                                check_apply_update_wrapper(
                                    u1.id(),   // the new row
                                    u2.id(),            // the new index to be added to row
                                    this_sim,                    // with this similarity
                                    &neighbour_is_new,
                                    &neighbourlarities,
                                    &work_done,
                                    &dao_manager,
                                );

                                check_apply_update_wrapper(
                                    u2.id(), // the new row
                                    u1.id(),           // the new index to be added to row
                                    this_sim,                   // with this similarity
                                    &neighbour_is_new,
                                    &neighbourlarities,
                                    &work_done,
                                    &dao_manager,
                                );
                            }
                        }
                    }
                },
            );

        let after = Instant::now();
        iterations += 1;

        log::debug!(
            "work done: {} num_data: {} iters: {}",
            work_done.load(std::sync::atomic::Ordering::SeqCst),
            num_data,
            iterations
        );
        log::debug!("Phase 2: {} ms", ((after - now).as_millis() as f64));
    }

    let final_time = Instant::now();
    log::trace!(
        "Total time for NN table build: {} ms",
        ((final_time - start_time).as_millis() as f64)
    );
}

// This code initialises the reverse links.
fn create_reverse_links<C: BitsContainer, const W: usize>(
    dao_manager: &DaoStore<C, { W }>,
    num_data: usize,
    neighbourlarities: &ArrayView2<Nality>,
    num_neighbours: usize,
    reverse_list_size: usize,
    old_neighbour_state: &mut ArrayView2<Nality>,
    old_flags: &ArrayView2<AtomicBool>,
) -> (Array2<Nality>, Array1<usize>) {
    let mut reverse_links: Array2<Nality> =
        Array2::from_elem((num_data, reverse_list_size), Nality::new_empty());
    let mut reverse_count = Array1::from_elem(num_data, 0);

    // loop over all current entries in neighbours;
    // add that entry to each row in the reverse list if that id is not in the forward NNs
    // there is a limit to the number of reverse ids we will store
    // so we will add only the most similar.

    for row in (0..num_data).map(|x| LocalAddress::into(x as u32)) {
        // Matlab line 97
        // all_ids are the forward links in the current id's row

        let this_row_neighbourlarities = &neighbourlarities.row(LocalAddress::as_usize(&row)); // Matlab line 98
                                                                                               // so for each one of these (there are k...):
        for id in 0..num_neighbours {
            // Matlab line 99 (updated)

            let this_global_id = this_row_neighbourlarities[id].id(); // get the id
                                                                      // and how similar it is to the current row id

            let this_sim = this_row_neighbourlarities[id].sim(); // may or may not be mapped

            if dao_manager.is_mapped(&this_global_id) {
                let this_local_id: LocalAddress = dao_manager
                    .table_addr_from_global_addr(&this_global_id)
                    .unwrap();

                let this_local_id: usize = LocalAddress::as_usize(&this_local_id);

                let new_forward_links = old_neighbour_state // get the previous neighbours of row row - can contain unmapped data
                    .row(this_local_id)
                    .iter()
                    .enumerate()
                    .filter_map(|(column, nality)| {
                        // and select the entries where previous_flags are set
                        if old_flags[[LocalAddress::as_usize(&row), column]].load(Ordering::Relaxed)
                        {
                            Some(nality.clone())
                        } else {
                            None
                        }
                    })
                    .collect::<Array1<Nality>>();

                // Check to see if the forward links contain the row under consideration already
                let forward_links_dont_contain_this_row = !new_forward_links.iter().any(|nality| {
                    nality.id().as_usize()          // A global address from current neighbours - may be mapped or unmapped
                        == dao_manager
                        .global_addr_from_table_addr(&row)
                        .unwrap()
                        .as_usize() // global address of row under consideration - always mapped
                });

                // if the reverse list isn't full, we will just add this one
                // this adds to a priority queue and keeps track of max

                // We are trying to find a set of reverse near neighbours with the
                // biggest similarity of size reverse_list_size.
                // first find all the forward links containing the row

                if forward_links_dont_contain_this_row {
                    // if the reverse list isn't full, we will just add this one
                    // this adds to a priority queue and keeps track of max

                    // We are trying to find a set of reverse near neighbours with the
                    // biggest similarity of size reverse_list_size.
                    // first find all the forward links containing the row

                    if reverse_count[this_local_id] < reverse_list_size {
                        // if the list is not full
                        // update the reverse pointer list and the similarities

                        reverse_links[[this_local_id, reverse_count[this_local_id]]] = Nality::new(
                            this_sim,
                            dao_manager.global_addr_from_table_addr(&row).unwrap(),
                        );
                        reverse_count[this_local_id] = reverse_count[this_local_id] + 1;
                    // increment the count
                    } else {
                        // the list is full - so no need to do anything with counts
                        // but it is, so we will only add it if it's more similar than another one already there

                        let (position, value) = min_index_and_value_neighbourlarities(
                            &reverse_links.row(this_local_id),
                        ); // Matlab line 109
                        let minimum_similarity_in_row = value.sim();

                        if minimum_similarity_in_row < this_sim {
                            // Matlab line 110  if the value in reverse_sims is less similar we overwrite
                            reverse_links[[this_local_id, position as usize]] = Nality::new(
                                this_sim,
                                dao_manager.global_addr_from_table_addr(&row).unwrap(),
                            );
                        }
                    }
                }
            }
        }
    }
    (reverse_links, reverse_count)
}

fn check_apply_update_wrapper<C: BitsContainer, const W: usize>(
    row_id: GlobalAddress,
    new_nality_addr: GlobalAddress,
    new_nality_similarity: f32,
    neighbour_is_new: &Array2<AtomicBool>,
    neighbourlarities: &Array2<Nality>,
    work_done: &AtomicUsize,
    dao_manager: &DaoStore<C, W>,
) {
    match dao_manager.table_addr_from_global_addr(&row_id) {
        // Only do the operation if the row is mapped
        Ok(row_id) => check_apply_update(
            LocalAddress::as_usize(&row_id),
            new_nality_addr.as_u32(),
            new_nality_similarity,
            neighbour_is_new,
            neighbourlarities,
            work_done,
        ),
        Err(_) => {}
    }
}

fn get_slice_using_multi_dao_selectors<C: BitsContainer, const W: usize>(
    dao_store: &DaoStore<C, W>,
    selectors: &ArrayView1<GlobalAddress>,
) -> Array1<EvpBits<C, W>> {
    let mut result = Array1::uninit(selectors.len());

    for count in 0..selectors.len() {
        let dao_holding_datum = dao_store
            .get_dao(&selectors[count])
            .unwrap_or_else(|_| panic!("Cannot find dao for addr {:?}", &selectors[count]));
        let source = &dao_holding_datum.get_data(); // the actual data indexed from zero
        let global_addr_selection = selectors[count]; // the global addr of the selection

        let pointer_to_evp_store =
            LocalAddress::into(global_addr_selection.as_u32() - dao_holding_datum.base_addr);

        source
            .slice(s![LocalAddress::as_usize(&pointer_to_evp_store)]) // assign the slice of evps to slot in result
            .assign_to(result.slice_mut(s![count]));
    }

    unsafe { result.assume_init() }
}

pub fn check_neighbours<C: BitsContainer, const W: usize>(
    neebs: &Array2<Nality>,
    dao_manager: &DaoStore<C, { W }>,
) {
    let mut count = 0;
    debug!("Checking neighbours");
    neebs.iter().for_each(|nality| {
        let global_addr: GlobalAddress = nality.id();
        count += 1;
        if !dao_manager.is_mapped(&global_addr) {
            debug!("Unmapped nality: {:?}", &global_addr);
            assert!(false);
        }
    });
    debug!("Finished checking {} neighbours", count);
}
