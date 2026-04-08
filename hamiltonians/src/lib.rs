/// Pascal's triangle lookup table.
/// Returns a 2D array of size (x rows) x (d cols),
/// analogous to MATLAB's makePascal_rel(x, d).
pub fn make_pascal_rel(rows: usize, cols: usize) -> Vec<Vec<f64>> {
    let mut tri = vec![vec![1f64; cols]; rows];
    // First row and first column are all 1s - set above.
    for i in 1..rows {
        for j in 1..cols {
            tri[i][j] = tri[i - 1][j] + tri[i][j - 1];
        }
    }
    tri
}

pub fn get_cycle_lengths_fast_rel(x: usize) -> Vec<usize> {
    let mut lengths = vec![0; x];

    for i in 1..=x {
        let ll = ((i as f64).log2().ceil()) as u32;
        lengths[i - 1] = 2_usize.pow(ll);
    }

    lengths
}

/// C(d+1-x, x+1) lookup from the Pascal table.
/// Panics if d+1-x < 1 (mirrors MATLAB's disp("oops")).
pub fn n_choose_k(d: usize, x: usize, tri: &[Vec<f64>]) -> f64 {
    assert!(d + 1 > x, "mchoosek_rel: oops: d = {d}, x = {x}");
    let row = d + 1 - x - 1; // -1 for 0-based indexing
    let col = x; // x+1-1 = x for 0-based indexing
    tri[row][col]
}

/// Returns the full initial arities vector for a given (x, d, arities) state.
/// Mirrors MATLAB's get_arities_full — mutates arities in place via the inner recursion.
pub fn get_arities_full(x: usize, d: usize, arities: &mut Vec<bool>) {
    fn do_level(level: usize, x: usize, d: usize, start_val: usize, arities: &mut Vec<bool>) {
        if level == x {
            // 0-based: level-1 in 0-based is level (since level starts at 1 in MATLAB)
            arities[level - 1] = !arities[level - 1];
        } else {
            let range_end = d - (x - level); // d - (x - level) inclusive
            let range: Vec<usize> = if arities[level - 1] {
                (start_val..=range_end).collect()
            } else {
                (start_val..=range_end).rev().collect()
            };

            for v in range {
                do_level(level + 1, x, d, v + 1, arities);
            }

            arities[level - 1] = !arities[level - 1];
        }
    }

    do_level(1, x, d, 1, arities);
}

/// Builds the cycle lookup table for a given x.
/// Returns a 2D Vec of shape [cycle_length,x].
/// Mirrors MATLAB's getCycleLookupTable_rel.
pub fn get_cycle_lookup_table(
    cycle_length: usize,
    x: usize,
    pascal: &[Vec<f64>],
) -> Vec<Vec<bool>> {
    let mut tab = vec![vec![]];

    // First row: get_arities_full(x, x, true(1,x))
    let mut first_row: Vec<bool> = vec![true; x];
    get_arities_full(x, x, &mut first_row);

    tab.push(first_row);

    for i in 1..cycle_length {
        let d = x + i;
        let mut alt_as = vec![false; x];
        for j in (1..=x).rev() {
            // MATLAB: j goes x:-1:1, index into altAs is end-j+1 = x-j (0-based)
            let nck = n_choose_k(d - j, x - j, pascal);
            alt_as[x - j] = nck % 2.0 == 0.0; // ~rem(nck,2)  //<<<<<<<<<<<<<< was /2 == 0 --- OK????
        }
        tab.push(alt_as);
    }

    return tab;
}

// was:
// pub fn get_cycle_lookup_table(
//     cycle_length: usize,
//     x: usize,
//     pascal: &[Vec<f64>],
// ) -> Vec<Vec<usize>> {
//     let mut tab = vec![vec![false; x]; cycle_length];
//
//     // First row: get_arities_full(x, x, true(1,x))
//     let mut first_row = vec![true; x];
//     get_arities_full(x, x, &mut first_row);
//     tab[0] = first_row;
//
//     for i in 1..cycle_length {
//         let d = x + i;
//         let mut alt_as = vec![false; x];
//         for j in (1..=x).rev() {
//             // MATLAB: j goes x:-1:1, index into altAs is end-j+1 = x-j (0-based)
//             let nck = n_choose_k(d - j, x - j, pascal);
//             alt_as[x - j] = nck % 2.0 == 0.0; // ~rem(nck,2)  //<<<<<<<<<<<<<< was /2 == 0 --- OK????
//         }
//         tab[i] = alt_as;
//     }
//
//     tab
// }

// Tables in the Matlab code is as follows:
// tables : 1 × N cell array
// tables{x} : logical matrix (2D)

/// Returns the new arities vector by XOR-ing with the cycle table row.
/// Mirrors MATLAB's getArities_rel.
pub fn get_arities(
    x: usize,
    d: usize,
    current_arities: &Vec<bool>,
    cycle_lengths: &[usize],
    tables: &Vec<Vec<Vec<bool>>>,
) -> Vec<bool> {
    let offset = (d - x) % cycle_lengths[x - 1]; // 0-based offset
    let this_table = &tables[x - 1];
    let cycle_row = &this_table[offset];
    current_arities
        .iter()
        .zip(cycle_row.iter())
        .map(|(&a, &c)| !(a ^ c)) // ~(xor(a, c))
        .collect()
}

pub fn get_path_index(
    x: usize,
    d: usize,
    vertex_val: &[bool],
    cycles: &Vec<usize>,
    tables: &Vec<Vec<Vec<bool>>>,
    pas_tri: &Vec<Vec<f64>>,
) -> f64 {
    fn get_inner_path_index(
        x: usize,
        d: usize,
        vertex_val: &[bool],
        arities: Vec<bool>,
        path_index: f64,
        cycles: &Vec<usize>,
        tables: &Vec<Vec<Vec<bool>>>,
        pas_tri: &Vec<Vec<f64>>,
    ) -> f64 {
        if x != d {
            // Case: arities is scalar (length == 1)
            if arities.len() == 1 {
                // This bit not needed since numbers are so big?
                // let set_bit = vertex_val
                //     .iter()
                //     .position(|&v| v)
                //     .map(|i| i + 1)
                //     .unwrap_or(0);
                // if arities[0] {
                //     return path_index + set_bit - 1;
                // } else {
                //     let set_bit_rev = vertex_val.len() - set_bit + 1;
                //     return path_index + set_bit_rev - 1;
                // }
                return path_index;
            } else {
                // Recursive case
                if vertex_val[0] {
                    if arities[0] {
                        // val(1) == 1, arities(1) == 1
                        return get_inner_path_index(
                            x - 1,
                            d - 1,
                            &vertex_val[1..],
                            arities[1..].to_vec(),
                            path_index,
                            cycles,
                            tables,
                            pas_tri,
                        );
                    } else {
                        // val(1) == 1, arities(1) == 0
                        let new_ars = get_arities(x, d - 1, &arities, cycles, tables);
                        return get_inner_path_index(
                            x - 1,
                            d - 1,
                            &vertex_val[1..],
                            new_ars[1..].to_vec(),
                            path_index + n_choose_k(d - 1, x, pas_tri),
                            cycles,
                            tables,
                            pas_tri,
                        );
                    }
                } else {
                    // val(1) == 0
                    if arities[0] {
                        let new_ars =
                            get_arities(x - 1, d - 1, &arities[1..].to_vec(), cycles, tables);

                        let mut combined = vec![arities[0]];
                        combined.extend(new_ars);

                        return get_inner_path_index(
                            x,
                            d - 1,
                            &vertex_val[1..],
                            combined,
                            path_index + n_choose_k(d - 1, x - 1, pas_tri),
                            cycles,
                            tables,
                            pas_tri,
                        );
                    } else {
                        // val(1) == 0, arities(1) == 0
                        return get_inner_path_index(
                            x,
                            d - 1,
                            &vertex_val[1..],
                            arities,
                            path_index,
                            cycles,
                            tables,
                            pas_tri,
                        );
                    }
                }
            }
        }

        path_index
    }

    // Initial call: true(1,x) → vec![true; x], pathIndex = 1
    get_inner_path_index(
        x,
        d,
        vertex_val,
        vec![true; x],
        1.0,
        cycles,
        tables,
        pas_tri,
    )
}

/// Top-level vertex number lookup.
/// Mirrors MATLAB's getVertexNumber_rel.
pub fn get_vertex_number(
    x: usize,
    d: usize,
    vertex: &[bool],
    cycle_lengths: &Vec<usize>,
    tables: &Vec<Vec<Vec<bool>>>,
    pas_tri: &Vec<Vec<f64>>,
) -> f64 {
    if vertex[0] {
        // vertex starts with 1: index into first part
        get_path_index(x - 1, d - 1, &vertex[1..], cycle_lengths, tables, pas_tri)
    } else {
        // vertex starts with 0: index into second part, offset by firstPartSize
        let first_part_size = n_choose_k(d - 1, x - 1, pas_tri);
        let reflected: Vec<bool> = vertex[1..].iter().rev().cloned().collect();
        let id = get_path_index(x, d - 1, &reflected, cycle_lengths, tables, pas_tri);
        id + first_part_size
    }
}
