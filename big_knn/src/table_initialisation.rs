use bits::container::Simd256x2;
use bits::evp::max_bsp_similarity_as_f32;
use ndarray::Array2;
use rand::Rng;
use std::time::Instant;
use utils::address::GlobalAddress;
use utils::Nality;

// This code is the same as the one in r_descent but uses GlobalAddress.

/// Initialise with a base address for the NN table
pub fn xxx_initialise_table_bsp_randomly(rows: usize, columns: usize) -> Array2<Nality> {
    log::info!("Randomly initializing table bsp, rows: {rows} neighbours: {columns}");
    let start_time = Instant::now();

    let mut rng = rand::rng();
    let nalities: Vec<Nality> = (0..rows * columns)
        .map(|_| {
            let rand_index = rng.random_range(0..rows); // pick random row index
            Nality::new_empty_sim(GlobalAddress::into(
                rand_index
                    .try_into()
                    .unwrap_or_else(|_| panic!("Cannot convert usize to u32")),
            ))
        })
        .collect();

    let mut nalities = Array2::from_shape_vec((rows, columns), nalities)
        .expect("Shape mismatch during initialisation");

    // overwrite first entry with a new nality of itself and 0
    for row in 0..nalities.nrows() {
        nalities[[row, 0]] = Nality::new(
            max_bsp_similarity_as_f32::<Simd256x2, 512>(),
            GlobalAddress::into(
                row.try_into()
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
