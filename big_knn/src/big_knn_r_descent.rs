use crate::dao_manager::{DaoManager, DaoStore};
// use crate::table_initialisation::initialise_table_bsp_randomly;
use crate::NalityNNTable;
use bits::container::BitsContainer;
use bits::evp::{matrix_dot, similarity_as_f32};
use bits::EvpBits;
use dao::Dao;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayViewMut1, Axis};
use r_descent::functions::{fill_false_atomic, fill_selected};
use r_descent::initialise_table_bsp_randomly;
use r_descent::{check_apply_update, get_slice_using_selectors, RDescent};
use rayon::prelude::*;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;
use utils::address::{GlobalAddress, LocalAddress};
use utils::{min_index_and_value_neighbourlarities, rand_perm, Nality};

pub fn into_big_knn_r_descent<C: BitsContainer, const W: usize>(
    daos: Vec<Dao<EvpBits<C, W>>>,
    num_neighbours: usize,
    reverse_list_size: usize,
    delta: f64,
    start_index: u32,
) -> NalityNNTable {
    let num_data = daos.iter().map(|dao| dao.num_data).sum();

    let neighbourlarities = initialise_table_bsp_randomly(num_data, num_neighbours, start_index);

    check_neighbours(&neighbourlarities, &daos[0]);

    make_big_knn_table2_bsp(
        daos,
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

fn check_neighbours<C: BitsContainer, const W: usize>(
    neebs: &Array2<Nality>,
    dao: &Dao<EvpBits<C, { W }>>,
) {
    println!("Checking neighbours");
    neebs.iter().for_each(|nality| {
        let id = GlobalAddress::as_u32(nality.id());
        if id < dao.base_addr && id > dao.base_addr + dao.embeddings.len() as u32 {
            println!("Unmapped nality: {:?}", &nality.id());
        }
    });
    println!("Finished checking neighbours");
}

pub fn make_big_knn_table2_bsp<C: BitsContainer, const W: usize>(
    daos: Vec<Dao<EvpBits<C, W>>>,
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

            // sampled are random indices from new_indices - they are indices into the current row

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

        for row in (0..num_data).map(|x| LocalAddress::into(x as u32)) {
            // Matlab line 97
            // all_ids are the forward links in the current id's row

            let this_row_neighbourlarities = &neighbourlarities.row(LocalAddress::as_usize(&row)); // Matlab line 98
                                                                                                   // so for each one of these (there are k...):
            for id in 0..num_neighbours {
                // Matlab line 99 (updated)
                // get the id
                let this_id = this_row_neighbourlarities[id].id();
                // and how similar it is to the current id
                let local_sim = this_row_neighbourlarities[id].sim();

                let this_local_id: LocalAddress =
                    dao_manager.table_addr_from_global_addr(&this_id).unwrap();

                let this_local_id: usize = LocalAddress::as_usize(&this_local_id);

                // newForwardLinks = new(thisId,:);
                let new_forward_links = new.row(this_local_id);

                // forwardLinksDontContainThis = sum(newForwardLinks == i_phase2) == 0;
                let forward_links_dont_contain_this = !new_forward_links
                    .iter()
                    .any(|x| dao_manager.table_addr_from_global_addr(&x.id()).unwrap() == row);

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

                        reverse[[this_local_id, reverse_count[this_local_id]]] = Nality::new(
                            local_sim,
                            dao_manager.global_addr_from_table_addr(&row).unwrap(),
                        );
                        reverse_count[this_local_id] = reverse_count[this_local_id] + 1;
                    // increment the count
                    } else {
                        // the list is full - so no need to do anything with counts
                        // but it is, so we will only add it if it's more similar than another one already there

                        let (position, value) =
                            min_index_and_value_neighbourlarities(&reverse.row(this_local_id)); // Matlab line 109
                        let value = value.sim();

                        if value < local_sim {
                            // Matlab line 110  if the value in reverse_sims is less similar we over write
                            reverse[[this_local_id, position as usize]] = Nality::new(
                                local_sim,
                                dao_manager.global_addr_from_table_addr(&row).unwrap(),
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

        old.axis_iter_mut(Axis(0)) // Get mutable rows (disjoint slices)
            .enumerate()
            .zip(new.axis_iter_mut(Axis(0))
            )
            .par_bridge()
            .map(|((row, old_row), new_row)| {

                    // println!( "old_row:" );
                    // old_row
                    //     .iter()
                    //     .map(|nality| { println!("{:?}", nality.id() ) });
                    // println!( "-----" );

                let binding = reverse
                    .row(row);

                let new_row_union: Array1<GlobalAddress> = if new_row.len() == 0 {
                    // Matlab line 130
                    Array1::from(vec![])
                } else {
                    new_row
                        .iter()
                        .filter_map(|x| { if x.is_empty() { None } else { Some(x.id()) } }) //<<<<<<<<< only take real values
                        .chain(binding
                            .iter()
                            .filter(|&x| !x.is_empty())
                            .map(|x| x.id()))
                        .collect::<Array1<GlobalAddress>>()
                };

                // index the data using the rows indicated in old_row
                let old_data = get_slice_using_multi_dao_selectors(
                    &dao_manager,
                    &old_row
                        .iter()
                        .map(|x| { x.id() })
                        .filter(|global_address: &GlobalAddress| dao_manager.is_mapped(*global_address, row)) // only look at addresses that are mapped - correct
                        .collect::<Array1<_>>().view()); // Matlab line 136

                if old_data.len() == 0 {
                    println!( "old data is empty!, old_row:" );
                    old_row
                        .iter()
                        .map(|nality| { println!("{:?}", nality.id() ) });
                    println!( "-----" );
                }

                let new_data = get_slice_using_multi_dao_selectors(
                    &dao_manager,
                    &new_row
                        .iter()
                        .map(|x| x.id())
                        .filter(|global_address: &GlobalAddress| dao_manager.is_mapped(*global_address, row)) // only look at addresses that are mapped  - correct
                        .collect::<Array1<_>>().view()); // Matlab line 137

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
                                    dao_manager.table_addr_from_global_addr(&u1_id).unwrap().as_usize(),
                                    GlobalAddress::as_u32(u2_id),
                                    this_sim,
                                    &neighbour_is_new,
                                    neighbourlarities,
                                    &work_done,
                                );
                                check_apply_update(
                                    dao_manager.table_addr_from_global_addr(&u2_id).unwrap().as_usize(),
                                    GlobalAddress::as_u32(u1_id),
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

                        if new_old_sims.shape() == &[18, 0] {
                            println!("new data shape = {:?}, old data shape = {:?}", new_data.shape(), old_data.shape());
                        }

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
                                    GlobalAddress::as_u32(u2.id()),            // <<<<<<<<<< the new index to be added to row
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
        log::debug!("Phase 3: {} ms", ((after - now).as_millis() as f64));
    }

    log::debug!(
        "Final iteration: c: {} iters: {}",
        work_done.load(std::sync::atomic::Ordering::SeqCst),
        iterations
    );

    let final_time = Instant::now();
    log::debug!(
        "Overall time 3: {} ms",
        ((final_time - start_time).as_millis() as f64)
    );
}

fn get_slice_using_multi_dao_selectors<C: BitsContainer, const W: usize>(
    dao_store: &DaoStore<C, W>,
    selectors: &ArrayView1<GlobalAddress>,
) -> Array1<EvpBits<C, W>> {
    let mut result = Array1::uninit(selectors.len());

    for count in 0..selectors.len() {
        let source = dao_store.get_dao(&selectors[count]).unwrap();
        let evps = &source.get_data(); // the actual data indexed from zero
        let global_addr_selection = selectors[count]; // the global addr of the selection
        let local_addr_selection = dao_store
            .table_addr_from_global_addr(&global_addr_selection)
            .unwrap(); // the corresponding index (indexed from zero)
        let local_addr_selection = local_addr_selection.as_usize(); // needs to be usize for index
        evps.slice(s![local_addr_selection]) // assign the slice of evps to slot in result
            .assign_to(result.slice_mut(s![count]));
    }

    unsafe { result.assume_init() }
}
