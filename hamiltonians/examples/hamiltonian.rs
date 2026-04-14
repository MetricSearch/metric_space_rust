// ---------------------------------------------------------------------------
// Example usage (mirrors the MATLAB example block)
// ---------------------------------------------------------------------------

use hamiltonians::{get_cycle_lengths, get_cycle_lookup_table, get_vertex_number, make_pascal};

fn example_usage() {
    let x: usize = 4;
    let d: usize = 8;

    // Build Pascal triangle
    let pas_tri: Vec<Vec<f64>> = make_pascal(d);

    // Build cycle lengths and lookup tables
    let cycle_lengths: Vec<usize> = get_cycle_lengths(d);

    println!("Cycle lengths - {:?}", cycle_lengths);

    let mut all_tables: Vec<Vec<Vec<bool>>> = vec![vec![vec![]]];

    for this_x in 1..=x {
        all_tables.push(get_cycle_lookup_table(cycle_lengths[x], this_x, &pas_tri));
    }

    println!("All tables:");
    for this_x in 1..=x {
        println!("{this_x}: entry -> {:?}", all_tables[this_x])
    }
    println!("End");

    let arities = vec![true; x - 1];

    // Example binary vertex (replace with real data)
    let vertex: Vec<bool> = vec![true, false, true, true, false, false, true, false];
    assert_eq!(vertex.len(), d);

    let result: f64 = get_vertex_number(x, d, vertex, &cycle_lengths, &all_tables, &pas_tri);
    println!("Vertex number: {result}");
}

fn main() {
    example_usage();
}
