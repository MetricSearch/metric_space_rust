//! Table initialisation code

use crate::functions::{
    get_slice_using_selected, insert_column_inplace, insert_index_at_position_1_inplace,
};
use crate::get_slice_using_selectors;
use bits::container::{BitsContainer, Simd256x2};
use bits::evp::max_bsp_similarity_as_f32;
use bits::{evp::matrix_dot, evp::similarity_as_f32, EvpBits};
use dao::Dao;
use ndarray::{s, Array2, Axis, Zip};
use rand::Rng;
use rayon::prelude::*;
use std::rc::Rc;
use std::time::Instant;
use utils::address::GlobalAddress;
use utils::{arg_sort_big_to_small_2d, bytes_fmt, rand_perm, Nality};

pub fn initialise_table_bsp_randomly_overwrite_row_0(
    rows: usize,
    columns: usize,
    start_index: u32,
) -> Array2<Nality> {
    log::info!("Randomly initializing table bsp, rows: {rows} neighbours: {columns}");
    let start_time = Instant::now();

    let mut rng = rand::rng();
    let nalities: Vec<Nality> = (0..rows * columns)
        .map(|_| {
            let rand_index = rng.random_range(0..rows); // pick random row index
            Nality::new_empty_sim(GlobalAddress::into(rand_index as u32 + start_index))
            // safe as range is bounded in previous line
        })
        .collect();

    let mut nalities = Array2::from_shape_vec((rows, columns), nalities)
        .expect("Shape mismatch during initialisation");

    // overwrite first entry with a new nality of itself and 0
    for row in 0..nalities.nrows() {
        nalities[[row, 0]] = Nality::new(
            max_bsp_similarity_as_f32::<Simd256x2, 512>(),
            GlobalAddress::into(
                (row as u32 + start_index)
                    .try_into()
                    .unwrap_or_else(|_| panic!("Cannot convert usize to u32")),
            ),
        );
    }

    let end_time = Instant::now();
    log::debug!(
        "Initialistion in {:?}ms",
        ((end_time - start_time).as_millis() as f64)
    );

    nalities
}

pub fn only_initialise_table_bsp_randomly(
    rows: usize,
    columns: usize,
    start_index: u32,
) -> Array2<Nality> {
    log::info!("Randomly initializing table bsp, rows: {rows} neighbours: {columns}");
    let start_time = Instant::now();

    let mut rng = rand::rng();
    let nalities: Vec<Nality> = (0..rows * columns)
        .map(|_| {
            let rand_index = rng.random_range(0..rows); // pick random row index
            Nality::new_empty_sim(GlobalAddress::into(rand_index as u32 + start_index))
            // safe as range is bounded in previous line
        })
        .collect();

    let mut nalities = Array2::from_shape_vec((rows, columns), nalities)
        .expect("Shape mismatch during initialisation");

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
    //let dims = dao.get_dim();
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
    let xx = Zip::from(&indices).and(&sims).map_collect(|&id, &sim| {
        Nality::new(
            sim,
            GlobalAddress::into(
                id.try_into()
                    .unwrap_or_else(|_| panic!("Cannot convert usize to u32")),
            ),
        )
    });

    // println!("First row initialisation: {:#?}", xx.row(0)); // TODO fix this

    xx
}
