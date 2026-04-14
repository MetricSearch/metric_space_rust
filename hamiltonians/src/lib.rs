/// Pascal's triangle lookup table.
/// Returns a 2D array of size (x rows) x (d cols),
/// analogous to MATLAB's makePascal_rel(x, d).
pub fn make_pascal(d: usize) -> Vec<Vec<f64>> {
    let mut tri = vec![vec![1f64; d]; d];
    // First row and first column are all 1s - set above.
    for i in 1..d {
        for j in 1..d {
            tri[i][j] = tri[i - 1][j] + tri[i][j - 1];
        }
    }
    tri
}

// produces a vector giving the cycle length for a given x therefore
// is indexed by x, a dummy value for x=0 is included to avoid indexing pain
pub fn get_cycle_lengths_original(x: usize) -> Vec<usize> {
    // the cycle length table is indexed by a semantic value x which is is [1,d]
    // so change from matlab by adding a dummy 0th value to the output which should never be accessed
    let mut lengths = vec![0; x + 1];

    for i in 1..=x {
        // in Python range(1,x + 1): - up to and including x
        // let ll = ((i as f64).log2().ceil()) as u32;
        //lengths[i] = 2_usize.pow(ll);

        let ll = ((i as f64).log2() / (2.0 as f64).log2()).ceil() as u32;
        lengths[i] = (2 as usize).pow(ll);
    }

    lengths
}

// produces a vector giving the cycle length for a given x therefore
// is indexed by x, a dummy value for x=0 is included to avoid indexing pain
// no floats and no overflow
pub fn get_cycle_lengths(x: usize) -> Vec<usize> {
    let mut lens = vec![0; x + 1];

    for i in 1..=x {
        let ll = (usize::BITS - (i - 1).leading_zeros()) as u32;
        lens[i] = 1usize << ll; // equivalent to 2^ll
    }

    lens
}

// Binomial
pub fn n_choose_k(d: usize, x: usize, tri: &Vec<Vec<f64>>) -> f64 {
    tri[d - x][x]
}

/// Returns the full initial arities vector for a given (x, d, arities) state.
/// return value includes an extra dummy field at the start
/// Mirrors MATLAB's get_arities_full — mutates arities in place via the inner recursion.
pub fn get_arities_full(x: usize, d: usize) -> Vec<bool> {
    fn do_level(
        x: usize,
        d: usize,
        level: usize,
        start_val: usize,
        prefix: Vec<usize>,
        arities: &mut Vec<bool>,
    ) {
        if level == x {
            arities[level] = !arities[level];
        } else {
            let range_start = start_val;
            let range_end = d - (x - level);

            if arities[level] {
                let range = range_start..range_end + 1;

                for v in range {
                    let mut new_prefix = prefix.clone();
                    new_prefix.push(v);
                    do_level(x, d, level + 1, v + 1, new_prefix, arities);
                }
            } else {
                let reversed_range = (range_start..range_end + 1).rev();

                for v in reversed_range {
                    let mut new_prefix = prefix.clone();
                    new_prefix.push(v);
                    do_level(x, d, level + 1, v + 1, new_prefix, arities);
                }
            };

            arities[level] = !arities[level];
        }
    }

    let mut arities = vec![true; x + 1]; // this has an extra dummy value in position 0.
    do_level(x, d, 1, 1, vec![], &mut arities);

    arities // return the
}

// for a given x, produces a table of derived arities for the next n values, where n is the
// length of the cycle. so table[0] gives all falses, the last row of the table reverts to that in the next iteration
pub fn get_cycle_lookup_table(
    cycle_length: usize,
    x: usize,
    pascal: &Vec<Vec<f64>>,
) -> Vec<Vec<bool>> {
    let mut tab = vec![vec![false; x]; cycle_length];

    for i in 1..cycle_length {
        let d = x + i;

        for j in (1..=x).rev() {
            let nck = n_choose_k(d - j, x - j, pascal) as usize; // will go fail if too big - but should not be
            tab[i][x - j] = nck % 2 == 0;
        }
    }

    tab
}

/// Returns the new arities vector by XOR-ing with the cycle table row.
/// Mirrors MATLAB's getArities_rel.
pub fn get_arities(
    x: usize,
    d: usize,
    current_arities: &Vec<bool>,
    cycle_length: usize,
    tables: &Vec<Vec<Vec<bool>>>,
) -> Vec<bool> {
    let this_table = &tables[x]; // NO DUMMY ROW IN TABLES - This is a Matrix
    let offset = (d - x) % cycle_length;
    let cycle_row = &this_table[offset];
    println!(
        "call of get_arities: x: {}, d: {}, current_arities: {:?} cycle_length: {:?} this_table: {:?}",
        x, d, current_arities, cycle_length, this_table
    );

    current_arities // this is  ~(currentArities ^ cycleRow) in Python
        .iter()
        .zip(cycle_row.iter())
        .map(|(a, b)| !(*a ^ *b))
        .collect::<Vec<bool>>()
}

// helper for argmax
fn argmax(v: Vec<bool>) -> usize {
    let mut max_idx = 0;
    let mut max_val = v[0];
    for (i, &val) in v.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }
    max_idx
}

pub fn get_vertex_number(
    x: usize,
    d: usize,
    vertex: Vec<bool>,
    cycles: &Vec<usize>,
    tables: &Vec<Vec<Vec<bool>>>,
    pas_tri: &Vec<Vec<f64>>,
) -> f64 {
    get_vertex_number_inner(x, d, vertex, &vec![true; x], 1f64, cycles, tables, pas_tri)
}

fn get_vertex_number_inner(
    x: usize,
    d: usize,
    vertex: Vec<bool>,
    arities: &Vec<bool>,
    previous_path_index: f64,
    cycles: &Vec<usize>,
    tables: &Vec<Vec<Vec<bool>>>,
    pas_tri: &Vec<Vec<f64>>,
) -> f64 {
    println!("*** d,x,v,ars: {d} {x} {:?} {:?}", vertex, arities);
    println!("Cycles length: {:?}", cycles.len());

    if x == d {
        return previous_path_index;
    }

    let new_path_index;

    let vertex_len = vertex.len() as f64;

    if arities.len() == 1 {
        let set_bit = argmax(vertex) as f64;

        if arities[0] {
            new_path_index = previous_path_index + set_bit;
        } else {
            let rev_pos = vertex_len - set_bit;
            new_path_index = previous_path_index + rev_pos - 1f64;
            println!(
                "increment (final): {}",
                new_path_index - previous_path_index
            );
        }
    } else {
        if vertex[0] {
            if arities[0] {
                println!("arm 1");
                // arm 1
                new_path_index = get_vertex_number_inner(
                    x - 1,
                    d - 1,
                    vertex[1..].to_vec(),
                    &arities[1..].to_vec(),
                    previous_path_index,
                    cycles,
                    tables,
                    pas_tri,
                );
            } else {
                println!("arm 2"); // arm 2
                println!("params to call arities: {x} {:?} {:?}", d - 1, arities);
                let new_ars = get_arities(x, d - 1, arities, cycles[x], tables);

                let increment = n_choose_k(d - 1, x, pas_tri);

                println!("increment {:?}", increment);

                new_path_index = get_vertex_number_inner(
                    x - 1,
                    d - 1,
                    vertex[1..].to_vec(),
                    &new_ars,
                    previous_path_index + increment,
                    cycles,
                    tables,
                    pas_tri,
                );
            }
        } else {
            if arities[0] {
                println!("arm 3"); // arm 3
                let new_ars = get_arities(x - 1, d - 1, &arities[1..].to_vec(), cycles[x], tables);
                println!("new_ars: {:?}", new_ars);
                println!("Shape of new_ars is {:?}", new_ars.len());

                let mut comp_ars = Vec::with_capacity(new_ars.len() + 1);
                comp_ars.push(arities[0]);
                comp_ars.extend(new_ars);

                println!("Comp ars: {:?}", comp_ars);
                let increment = n_choose_k(d - 1, x - 1, pas_tri);
                println!("increment: {:?}", increment);

                new_path_index = get_vertex_number_inner(
                    x,
                    d - 1,
                    vertex[1..].to_vec(),
                    &comp_ars,
                    previous_path_index + increment,
                    cycles,
                    tables,
                    pas_tri,
                );
            } else {
                println!("arm 4"); // arm 4
                new_path_index = get_vertex_number_inner(
                    x,
                    d - 1,
                    vertex[1..].to_vec(),
                    arities,
                    previous_path_index,
                    cycles,
                    tables,
                    pas_tri,
                );
            }
        }
    }

    new_path_index
}
