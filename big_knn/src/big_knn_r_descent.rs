use crate::dao_manager::{get_ranges, DaoManager, DaoStore};
// use crate::table_initialisation::initialise_table_bsp_randomly;
use crate::NalityNNTable;
use bits::container::{BitsContainer, Simd256x2};
use bits::evp::{matrix_dot, similarity_as_f32};
use bits::EvpBits;
use dao::Dao;
use log::debug;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayViewMut1, Axis, CowArray};
use r_descent::functions::{fill_false_atomic_from_selectors, fill_selected};
use r_descent::initialise_table_bsp_randomly;
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
    let num_data = daos.iter().map(|dao| dao.num_data).sum();

    let neighbourlarities = initialise_table_bsp_randomly(num_data, num_neighbours, start_index);

    let dao_manager = DaoStore::new(daos);

    check_neighbours(&neighbourlarities, &dao_manager);

    make_big_knn_table2_bsp(
        dao_manager,
        num_data,
        &neighbourlarities,
        num_neighbours,
        delta,
        reverse_list_size,
    );

    let ords = neighbourlarities.mapv(|x| GlobalAddress::as_usize(x.id()));
    let dists = neighbourlarities.mapv(|x| x.sim() as f32);

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
    log::debug!("Neighbourities length: {:?}", neighbourlarities.shape(),);

    let start_time = Instant::now();

    let mut iterations = 0;

    let mut neighbour_is_new: Array2<AtomicBool> =
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

        // Take a copy of the state of the world
        let mut previous_neighbours = &mut neighbourlarities.clone();
        let previous_flags: Array2<AtomicBool> =
            Array2::from_shape_fn(neighbour_is_new.dim(), |(i, j)| {
                // clone the atomic flags array
                AtomicBool::new(neighbour_is_new[(i, j)].load(Ordering::Relaxed))
            });

        // Phase 1 (was phase 2)  Matlab line 88

        let now = Instant::now();

        let mut reverse_links: Array2<Nality> =
            Array2::from_elem((num_data, reverse_list_size), Nality::new_empty());
        // reverse_ptr - how many reverse pointers for each entry in the dataset
        let mut reverse_count = Array1::from_elem(num_data, 0);

        // loop over all current entries in neighbours; add that entry to each row in the
        // reverse list if that id is in the forward NNs
        // there is a limit to the number of reverse ids we will store, as these
        // are in a zipf distribution, so we will add the most similar only

        for row in (0..num_data).map(|x| LocalAddress::into(x as u32)) {
            // Matlab line 97
            // all_ids are the forward links in the current id's row

            let this_row_neighbourlarities = &neighbourlarities.row(LocalAddress::as_usize(&row)); // Matlab line 98
                                                                                                   // so for each one of these (there are k...):
            for id in 0..num_neighbours {
                // Matlab line 99 (updated)
                // get the id
                let this_global_id = this_row_neighbourlarities[id].id();
                // and how similar it is to the current id
                let this_sim = this_row_neighbourlarities[id].sim();

                if dao_manager.is_mapped(this_global_id) {
                    let this_local_id: LocalAddress = dao_manager
                        .table_addr_from_global_addr(&this_global_id)
                        .unwrap();

                    let this_local_id: usize = LocalAddress::as_usize(&this_local_id);

                    let new_forward_links = previous_neighbours //take the previous neighbours
                        .row(this_local_id)
                        .iter()
                        .enumerate()
                        .filter_map(|(column, nality)| {
                            // and select the entries where previous_flags are set
                            if previous_flags[[LocalAddress::as_usize(&row), column]]
                                .load(Ordering::Relaxed)
                            {
                                Some(nality.clone())
                            } else {
                                None
                            }
                        })
                        .collect::<Array1<Nality>>();

                    let forward_links_dont_contain_this = !new_forward_links.iter().any(|nality| {
                        GlobalAddress::as_usize(nality.id())
                            == GlobalAddress::as_usize(
                                dao_manager.global_addr_from_table_addr(&row).unwrap(),
                            )
                        // safe this way around - rows are all mapped - TODO check other similar operations?
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

                        if reverse_count[this_local_id] < reverse_list_size {
                            // if the list is not full
                            // update the reverse pointer list and the similarities

                            reverse_links[[this_local_id, reverse_count[this_local_id]]] =
                                Nality::new(
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
                            let value = value.sim();

                            if value < this_sim {
                                // Matlab line 110  if the value in reverse_sims is less similar we over write
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

        let after = Instant::now();
        log::debug!("Phase 1: {} ms", ((after - now).as_millis() as f64));

        // Phase 2 (was phase 3)

        let now = Instant::now();

        work_done = AtomicUsize::new(0);

        previous_neighbours
            .axis_iter(Axis(0)) // iterate over the rows
            .enumerate()
            //.par_bridge() // TODO put back par_bridge
            .map(|(row_index, nalities)| {
                let previous_row = previous_neighbours.row(row_index);

                let new_row: Vec<_> = previous_row
                    .into_iter()
                    .enumerate()
                    .filter(|(column, nality)| previous_flags[[row_index, *column]].load(Ordering::Relaxed))
                    .map(|x| x.1)
                    .collect();

                let old_row: Vec<_> = previous_row
                    .into_iter()
                    .enumerate()
                    .filter(|(column, nality)| ! previous_flags[[row_index, *column]].load(Ordering::Relaxed))
                    .map(|x| x.1)
                    .collect();

                let reverse_row_links = reverse_links.row(row_index); // may contain unmapped data?

                let new_mapped_forward_and_reverse_links: Array1<GlobalAddress> = if new_row.len() == 0 {
                    // Matlab line 130
                    Array1::from(vec![])
                } else {
                    new_row
                        .iter()
                        .filter_map(|x| { if x.is_empty() || ! dao_manager.is_mapped(x.id()) { None } else { Some(x.id()) } }) // only take mapped values
                        .chain(reverse_row_links
                            .iter()
                            .filter(|&x| !x.is_empty() && dao_manager.is_mapped(x.id()) ) // only take mapped values
                            .map(|x| x.id()))
                        .collect::<Array1<GlobalAddress>>()
                };

                // index the data using the rows indicated in old_row
                let old_mapped_row_data = get_slice_using_multi_dao_selectors( // an array of evps selected from the old row for mapped entities
                                                                               &dao_manager,
                                                                               &old_row
                                                                        .iter()
                                                                        .map(|x| { x.id() })
                                                                        .filter(|global_address: &GlobalAddress| dao_manager.is_mapped(*global_address)) // only look at addresses that are mapped
                                                                        .collect::<Array1<GlobalAddress>>().view()); // Matlab line 136

                let new_mapped_row_data = get_slice_using_multi_dao_selectors(
                    &dao_manager,
                    &new_row // may contain data that is mapped and unmapped
                        .iter()
                        .map(|x| x.id())
                        .filter(|global_address: &GlobalAddress| dao_manager.is_mapped(*global_address)) // only look at addresses that are mapped
                        .collect::<Array1<GlobalAddress>>().view()); // Matlab line 137

                let new_mapped_forward_and_reverse_data = // all elements of new_row_union are mapped
                     get_slice_using_multi_dao_selectors(&dao_manager, &new_mapped_forward_and_reverse_links.view()); // Matlab line 137

                let new_mapped_forward_and_reverse_sims =
                    matrix_dot(new_mapped_forward_and_reverse_data.view(), new_mapped_forward_and_reverse_data.view(), |a, b| {
                        similarity_as_f32(a, b)
                    });

                (
                    new_row,
                    old_row,
                    new_mapped_forward_and_reverse_links,
                    new_mapped_forward_and_reverse_sims,
                    new_mapped_row_data,
                    old_mapped_row_data,
                )
            })
            .for_each(
                |(new_row,
                     old_row,
                     new_mapped_forward_and_reverse_links,
                     new_mapped_forward_and_reverse_sims,
                     new_mapped_row_data,
                     old_mapped_row_data,
                 )| {
                    // Two for loops for the two distance tables (similarities and new_old_sims) for each pair of elements in the newNew list, their original ids
                    // First iterate over new_new_sims.. upper triangular (since distance table)

                    if new_mapped_forward_and_reverse_links.len() >= 2 {
                        // must be at least 2 elements in the array because we are doing pair-wise comparisons.

                        for new_ind1 in 0..new_mapped_forward_and_reverse_links.len() - 1 {
                            // Matlab line 144 (-1 since don't want the diagonal)
                            let u1_id = *new_mapped_forward_and_reverse_links.get(new_ind1).unwrap_or_else(|| panic!("Illegal index of new_row_union at {new_ind1} length is: {}", new_mapped_forward_and_reverse_links.len()));

                            for new_ind2 in new_ind1 + 1..new_mapped_forward_and_reverse_links.len() {
                                // Matlab line 147
                                let u2_id = *new_mapped_forward_and_reverse_links.get(new_ind2).unwrap_or_else(|| panic!("Illegal index of new_row_union at {new_ind2} length is: {}", new_mapped_forward_and_reverse_links.len()));
                                // then get their similarity from the matrix
                                let this_sim = *new_mapped_forward_and_reverse_sims.get((new_ind1, new_ind2)).unwrap_or_else(|| panic!("Illegal index of new_new_sims at {new_ind1},{new_ind2} Shape is: {:?}", new_mapped_forward_and_reverse_sims.shape()));
                                // is the current similarity greater than the biggest distance
                                // in the row for u1_id? if it's not, then do nothing


                                let u1_row_id = dao_manager.table_addr_from_global_addr(&u1_id).unwrap();
                                let u2_row_id = dao_manager.table_addr_from_global_addr(&u2_id).unwrap();


                                check_apply_update_wrapper(
                                    dao_manager.table_addr_from_global_addr(&u1_id).unwrap(),
                                    u2_id,
                                    this_sim,
                                    &neighbour_is_new,
                                    neighbourlarities,
                                    &work_done,
                                );

                                check_apply_update_wrapper(
                                        dao_manager.table_addr_from_global_addr(&u2_id).unwrap(),
                                        u1_id,
                                        this_sim,
                                        &neighbour_is_new,
                                        neighbourlarities,
                                        &work_done,
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

                        // and do the same for each pair of elements in the new_row/old_row

                        for new_row_index_1 in 0..new_row.len() {
                            // Matlab line 183  // rectangular matrix - need to look at all

                            let u1 = &new_row.get(new_row_index_1).unwrap_or_else(|| panic!("Illegal index of new_row at {new_row_index_1} length is: {}", new_row.len()));
                            for new_row_index_2 in 0..old_row.len() {
                                let u2 = &old_row.get(new_row_index_2).unwrap(); // Matlab line 186

                                // then get their distance from the matrix

                                let this_sim = *new_old_sims.get((new_row_index_1, new_row_index_2))
                                    .unwrap_or_else(|| panic!("Illegal index of new_old_sims at {new_row_index_1},{new_row_index_2} Shape is: {:?}", new_old_sims.shape()));

                                check_apply_update(
                                    dao_manager.table_addr_from_global_addr(&u1.id()).unwrap().as_usize(),   // the new row
                                    GlobalAddress::as_u32(u2.id()),            // the new index to be added to row
                                    this_sim,           // with this similarity
                                    &neighbour_is_new,
                                    neighbourlarities,
                                    &work_done,
                                );

                                check_apply_update(
                                    dao_manager.table_addr_from_global_addr(&u2.id()).unwrap().as_usize(),
                                    GlobalAddress::as_u32(u1.id()),
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
        log::debug!("Phase 2: {} ms", ((after - now).as_millis() as f64));

        println!("Row 0: {:?}", neighbourlarities.row(0));
    }

    log::debug!(
        "Final iteration: c: {} iters: {}",
        work_done.load(std::sync::atomic::Ordering::SeqCst),
        iterations
    );

    let final_time = Instant::now();
    log::debug!(
        "Overall time: {} ms",
        ((final_time - start_time).as_millis() as f64)
    );
}

fn check_apply_update_wrapper(
    row_id: LocalAddress,
    new_nality_addr: GlobalAddress,
    new_nality_similarity: f32,
    neighbour_is_new: &Array2<AtomicBool>,
    neighbourlarities: &Array2<Nality>,
    work_done: &AtomicUsize,
) {
    check_apply_update(
        LocalAddress::as_usize(&row_id),
        GlobalAddress::as_u32(new_nality_addr),
        new_nality_similarity,
        neighbour_is_new,
        neighbourlarities,
        work_done,
    );
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

        let local_offset = dao_store
            .table_addr_from_global_addr(&global_addr_selection)
            .unwrap(); // TODO very inefficient - replace with below.
                       // GlobalAddress::as_usize(global_addr_selection) - dao_holding_datum.base_addr as usize;

        source
            .slice(s![LocalAddress::as_usize(&local_offset)]) // assign the slice of evps to slot in result
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
        if !dao_manager.is_mapped(global_addr) {
            debug!("Unmapped nality: {:?}", &global_addr);
            assert!(false);
        }
    });
    debug!("Finished checking {} neighbours", count);
}
