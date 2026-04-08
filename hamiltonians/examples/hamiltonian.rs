// ---------------------------------------------------------------------------
// Example usage (mirrors the MATLAB example block)
// ---------------------------------------------------------------------------

use hamiltonians::{
    get_cycle_lengths_fast_rel, get_cycle_lookup_table, get_vertex_number, make_pascal_rel,
};

fn example_usage() {
    let x: usize = 4; // small example; MATLAB used 192
    let d: usize = 8; // MATLAB used 384

    // Build Pascal triangle
    let pas_tri = make_pascal_rel(d + 2, d + 2);

    // Build cycle lengths and lookup tables
    let c_lengths = get_cycle_lengths_fast_rel(x);
    let mut tables: Vec<Vec<Vec<bool>>> = Vec::with_capacity(x);
    for xi in 1..=x {
        tables.push(get_cycle_lookup_table(c_lengths[xi - 1], xi, &pas_tri));
    }

    // Example binary vertex (replace with real data)
    let vertex: Vec<bool> = vec![true, false, true, true, false, false, true, false];
    assert_eq!(vertex.len(), d);

    let result = get_vertex_number(x, d, &vertex, &c_lengths, &tables, &pas_tri);
    println!("Vertex number: {result}");
}

fn main() {
    example_usage();
}
