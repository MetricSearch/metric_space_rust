//! Table initialisation Hamiltonian code

//use bits::container::BitsContainer;
use half::f16;
use hamiltonians::{get_cycle_lengths, get_cycle_lookup_table, get_vertex_number, make_pascal};
use ndarray::{s, stack, Array1, Array2, ArrayView, ArrayView1, Axis, Ix2};
use rayon::prelude::*;
use utils::non_nan::NonNan;
use utils::non_nan_f64::NonNanF64;

pub fn hamiltonian_table_init(data_f16: ArrayView1<Array1<f16>>) -> Array2<usize> {
    //----- First Set up Hamiltonian machinery

    //let rows = data_f16.len();
    let columns = data_f16[0].len();

    let D = columns;
    let x = columns / 2;

    // Build Pascal triangle
    let pas_tri: Vec<Vec<f64>> = make_pascal(D);
    // Build cycle lengths and lookup tables
    let cycle_lengths: Vec<usize> = get_cycle_lengths(D);
    let mut all_tables: Vec<Vec<Vec<bool>>> = vec![vec![vec![]]];

    // TODO make these smaller if we are building multiple NN tables with less width than D.

    for this_x in 1..=x {
        all_tables.push(get_cycle_lookup_table(cycle_lengths[x], this_x, &pas_tri));
    }

    let code_size = 26; // The number of levels in the simplex rep too big but OK - TODO make this something more reasoned

    const NUM_NEIGHBOURS: usize = 15;

    let slice1 = data_f16
        .iter() // slice out the first code size columns
        .map(|row| row.slice(s![..code_size]))
        .collect::<Array1<ArrayView1<f16>>>();

    let (nn_table1, dists1) = get_nn_table(
        NUM_NEIGHBOURS,
        &slice1,
        code_size / 2,
        code_size,
        &cycle_lengths,
        &all_tables,
        &pas_tri,
    );

    let slice2 = data_f16
        .iter() // slice the next code size columns
        .map(|row| row.slice(s![code_size..code_size * 2]))
        .collect::<Array1<ArrayView1<f16>>>();

    let (nn_table2, dists2) = get_nn_table(
        NUM_NEIGHBOURS,
        &slice2,
        code_size / 2,
        code_size,
        &cycle_lengths,
        &all_tables,
        &pas_tri,
    );

    let nns: Array2<usize> = merge_tables(&nn_table1, &dists1, &nn_table2, &dists2, NUM_NEIGHBOURS);

    nns
}

fn merge_tables(
    nn_table1: &Array2<usize>,
    dists1: &Array2<f32>,
    nn_table2: &Array2<usize>,
    dists2: &Array2<f32>,
    required_size: usize,
) -> Array2<usize> {
    assert_eq!(nn_table1.dim(), nn_table2.dim());
    assert_eq!(dists1.dim(), dists2.dim());
    assert_eq!(nn_table1.dim(), dists1.dim());

    let (n, k) = nn_table1.dim();
    assert!(required_size <= 2 * k);

    let mut merged_nns = Array2::<usize>::zeros(((n, required_size)));

    for row in 0..n {
        // bind views first (fixes temporary issue)
        let row1_d_view = dists1.row(row);
        let row2_d_view = dists2.row(row);
        let row1_i_view = nn_table1.row(row);
        let row2_i_view = nn_table2.row(row);

        // then take slices
        let row1_d = row1_d_view.as_slice().expect("contiguous");
        let row2_d = row2_d_view.as_slice().expect("contiguous");
        let row1_i = row1_i_view.as_slice().expect("contiguous");
        let row2_i = row2_i_view.as_slice().expect("contiguous");

        let mut merged_row_view = merged_nns.row_mut(row);
        let merged_row = merged_row_view.as_slice_mut().unwrap();

        let mut p1 = 0;
        let mut p2 = 0;

        for output in 0..required_size {
            let take1 = p1 < k && (p2 >= k || row1_d[p1] <= row2_d[p2]);

            if take1 {
                merged_row[output] = row1_i[p1];
                p1 += 1;
            } else {
                merged_row[output] = row2_i[p1];
                p2 += 1;
            }
        }
    }

    merged_nns
}

fn get_nn_table(
    num_neighbours: usize,
    data_f16: &Array1<ArrayView1<f16>>,
    x: usize,
    d: usize,
    cycles: &Vec<usize>,
    tables: &Vec<Vec<Vec<bool>>>,
    pas_tri: &Vec<Vec<f64>>,
) -> (Array2<usize>, Array2<f32>) {
    let data_size = data_f16.len();

    let hamiltonian_values_in_data_order =
        get_hamiltonians_in_data_order(&data_f16, x, d, &cycles, &tables, &pas_tri, data_size);

    // hamiltonian_order_to_data_order maps from hamiltonian order to data order

    let (hamiltonian_order_to_data_order, _) =
        arg_sort_small_to_big_1d_f64(&hamiltonian_values_in_data_order);

    let hamiltonian_order_to_data_order: Array1<usize> =
        Array1::from(hamiltonian_order_to_data_order); // retype the indices so we can use views over the window

    const window_size: usize = 1000;

    let mut slice_start = 0;

    // create a nn table
    let mut nn_table: Array2<usize> = Array2::<usize>::zeros((data_size, num_neighbours));
    let mut dists_table: Array2<f32> = Array2::<f32>::zeros((data_size, num_neighbours));

    // initialise the nn table a window_size elements at a time
    // takes the smallest distances within this window
    // TODO could deal with the start/end of the cycle better

    // TODO Make this parallel.

    for selector_slice in hamiltonian_order_to_data_order.axis_chunks_iter(Axis(0), window_size) {
        let view_f32_embeddings: Array1<Array1<f32>> = data_f16
            .select(Axis(0), selector_slice.as_slice().unwrap())
            .mapv(|x| x.mapv(|x| x.to_f32()));

        let owned_f32_embeddings: Array2<f32> = stack(
            Axis(0),
            &view_f32_embeddings
                .iter()
                .map(|r| r.view())
                .collect::<Vec<_>>(),
        )
        .unwrap();

        // Find the  distances over data_values
        let chunk_dists = owned_f32_embeddings.dot(&owned_f32_embeddings.t()); // dists in snake order over slice

        // get the first num_neighbours out of the sorted_ords and assign to the nn table.
        // the ordered_indices are relative to the range from 0..window_size to get to data relative add the slice_start.
        // the nns are in the natural order

        for row in 0..window_size {
            let (row_window_relative_position_in_hamiltonian_order, dists) =
                arg_sort_big_to_small_1d_f32(chunk_dists.row(row)); // sorted ords are relative (to the window) indices

            for col in 0..num_neighbours {
                let absolute_position_hamiltonian_order =
                    row_window_relative_position_in_hamiltonian_order[col] + slice_start; // index into the data
                nn_table[[row + slice_start, col]] =
                    hamiltonian_order_to_data_order[absolute_position_hamiltonian_order];
                dists_table[[row + slice_start, col]] = dists[col]; // distance of the corresponding entry
            }
        }

        slice_start += window_size;
    }

    print!("First 2 lines of approx NNS: ");
    for row in 0..2 {
        for i in 0..num_neighbours {
            print!("{} ", nn_table[[row, i]]);
        }
        println!();
    }

    print!("First 2000,2001 lines of approx NNS: ");
    for row in 0..2 {
        for i in 0..num_neighbours {
            print!("{} ", nn_table[[row + 2000, i]]);
        }
        println!();
    }

    (nn_table, dists_table)
}

fn get_hamiltonians_in_data_order(
    data_f16: &Array1<ArrayView1<f16>>,
    x: usize,
    D: usize,
    cycles: &Vec<usize>,
    tables: &Vec<Vec<Vec<bool>>>,
    pas_tri: &Vec<Vec<f64>>,
    data_size: usize,
) -> Vec<f64> {
    // hamiltonian_values_in_data_order map from original data index to Hamiltonian f64 vertex values
    let mut hamiltonian_values_in_data_order = Vec::with_capacity(data_size);
    // Turn the f16 data into Hamiltonian f64 vertex numbers and add to snake_positions_from_data data structure
    for vertex in data_f16 {
        let vertex: Vec<bool> = f16_vec_to_bool_vec(vertex);
        let v_no: f64 = get_vertex_number(x, D, vertex, &cycles, &tables, &pas_tri);
        hamiltonian_values_in_data_order.push(v_no);
    }
    hamiltonian_values_in_data_order
}

fn median(arrai: &ArrayView1<f16>) -> f32 {
    let mut v = arrai.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let len = v.len();

    if len % 2 == 1 {
        v[len / 2].to_f32()
    } else {
        (v[len / 2 - 1] + v[len / 2]).to_f32() / 2.0
    }
}

// TODO these functions below should be in utils somewhere - perhaps some are? - check.

fn f16_vec_to_bool_vec(arrai: &ArrayView1<f16>) -> Vec<bool> {
    let median: f32 = median(arrai);
    arrai.iter().map(|&x| (x.to_f32()) < median).collect()
}

fn arg_sort_small_to_big_2d_f32(vals: &ArrayView<f32, Ix2>) -> (Vec<usize>, Vec<f32>) {
    let mut enumerated = vals.iter().enumerate().collect::<Vec<(usize, &f32)>>(); // Vec of positions (ords) and values (dists)
    enumerated.sort_by(|a, b| NonNan::new(*a.1).partial_cmp(&NonNan::new(*b.1)).unwrap());
    enumerated.into_iter().unzip()
}

fn arg_sort_small_to_big_1d_f64(vals: &Vec<f64>) -> (Vec<usize>, Vec<f64>) {
    let mut enumerated = vals.iter().enumerate().collect::<Vec<(usize, &f64)>>(); // Vec of positions (ords) and values (dists)
    enumerated.sort_by(|a, b| {
        NonNanF64::new(*a.1)
            .partial_cmp(&NonNanF64::new(*b.1))
            .unwrap()
    });
    enumerated.into_iter().unzip()
}

fn arg_sort_small_to_big_1d_f32(vals: ArrayView1<f32>) -> (Vec<usize>, Vec<f32>) {
    let mut enumerated = vals.iter().enumerate().collect::<Vec<(usize, &f32)>>(); // Vec of positions (ords) and values (dists)
    enumerated.sort_by(|a, b| NonNan::new(*a.1).partial_cmp(&NonNan::new(*b.1)).unwrap());
    enumerated.into_iter().unzip()
}

fn arg_sort_big_to_small_1d_f32(vals: ArrayView1<f32>) -> (Vec<usize>, Vec<f32>) {
    let mut enumerated = vals.iter().enumerate().collect::<Vec<(usize, &f32)>>(); // Vec of positions (ords) and values (dists)
    enumerated.sort_by(|a, b| NonNan::new(*b.1).partial_cmp(&NonNan::new(*a.1)).unwrap());
    enumerated.into_iter().unzip()
}
