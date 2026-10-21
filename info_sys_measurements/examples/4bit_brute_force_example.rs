use anyhow::Result;
use ndarray::Array1;
use rand::random;
use std::time::Instant;
use wide::{i16x16, i32x8, u8x16, u8x32};

pub fn euc_4bit(a: &[u8x32], b: &[u8x32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = i32x8::ZERO;
    let mask = i16x16::splat(15);

    for (a, b) in a.iter().zip(b.iter()) {
        for (a, b) in a
            .as_array_ref()
            .chunks_exact(16)
            .zip(b.as_array_ref().chunks_exact(16))
        {
            // Widen packed bytes before extracting their unsigned nibbles.
            let a = i16x16::from(u8x16::from(a));
            let b = i16x16::from(u8x16::from(b));
            let low_diff = (a & mask) - (b & mask);
            let high_diff = (a >> 4_u32) - (b >> 4_u32);

            sum += low_diff.dot(low_diff);
            sum += high_diff.dot(high_diff);
        }
    }

    (sum.reduce_add() as f32).sqrt()
}

fn to_packed_4bit_vectors(array: &Array1<u8>, dims: usize) -> Vec<u8x32> {
    assert!(dims > 0);
    assert_eq!(array.len() % dims, 0);
    debug_assert!(array.iter().all(|&x| x <= 15));

    array
        .as_slice()
        .expect("array must be contiguous")
        .chunks_exact(dims)
        .flat_map(|vector| {
            // One 256-bit block holds 64 dimensions, two per byte.
            vector.chunks(64).map(|chunk| {
                let mut packed = [0; 32];
                for (byte, pair) in packed.iter_mut().zip(chunk.chunks(2)) {
                    *byte = pair[0] | (pair.get(1).copied().unwrap_or(0) << 4);
                }
                u8x32::new(packed)
            })
        })
        .collect()
}

pub fn to_u4_array(array: &Array1<f32>, max_f32: f32) -> Array1<u8> {
    array.mapv(|x| {
        let value = x / max_f32;
        if value.is_nan() {
            0
        } else {
            (value * 15.0).round().clamp(0.0, 15.0) as u8
        }
    })
}

fn main() -> Result<()> {
    let num_data = 1_000_000;

    //----------------

    for dims in [100, 384, 500, 768] {
        do_experiment(num_data, dims)
    }

    Ok(())
}

fn do_experiment(num_data: usize, dims: usize) {
    let query_f32 = Array1::from_iter((0..dims).map(|_| random::<f32>()));
    let data_f32 = Array1::from_iter((0..dims * num_data).map(|_| random::<f32>()));

    let max_f32 = data_f32
        .iter()
        .copied()
        .map(|x| x.abs())
        .fold(f32::NEG_INFINITY, f32::max);

    let query = to_u4_array(&query_f32, max_f32);
    let data = to_u4_array(&data_f32, max_f32);

    let query_wide = to_packed_4bit_vectors(&query, dims);
    let data_wide = to_packed_4bit_vectors(&data, dims);

    let now = Instant::now();

    // Do a brute force of queries against the data
    let four_bit_distances = generate_4bit_dists(&query_wide, &data_wide, dims);

    let after = Instant::now();
    eprintln!("Last distance is {:?}", four_bit_distances.iter().last());

    println!(
        "Int4 {}:\t{} ns",
        dims,
        ((after - now).as_nanos() as f64) as f64
    );
}

fn generate_4bit_dists(query: &Vec<u8x32>, data: &Vec<u8x32>, dims: usize) -> Vec<f32> {
    assert!(dims > 0);
    let blocks_per_vector = dims.div_ceil(64);
    assert_eq!(query.len(), blocks_per_vector);
    assert_eq!(data.len() % blocks_per_vector, 0);

    data.chunks_exact(blocks_per_vector)
        .map(|d| euc_4bit(query, d))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantization_uses_the_unsigned_four_bit_range() {
        let input = Array1::from_vec(vec![
            -1.0,
            0.0,
            1.0 / 15.0,
            0.5,
            14.0 / 15.0,
            1.0,
            2.0,
            f32::NAN,
        ]);
        assert_eq!(
            to_u4_array(&input, 1.0),
            Array1::from_vec(vec![0, 0, 1, 8, 14, 15, 15, 0])
        );
        assert_eq!(
            to_u4_array(&Array1::from_vec(vec![0.0, 1.0, 2.0]), 2.0),
            Array1::from_vec(vec![0, 8, 15])
        );
        assert_eq!(to_u4_array(&Array1::zeros(3), 0.0), Array1::zeros(3));
    }

    #[test]
    fn every_nibble_pair_matches_scalar_distance() {
        let data = Array1::from_iter((0..=255_u8).flat_map(|byte| [byte & 15, byte >> 4]));
        let packed = to_packed_4bit_vectors(&data, 2);
        for byte in 0..=255_u8 {
            let query = Array1::from_vec(vec![byte & 15, byte >> 4]);
            let query = to_packed_4bit_vectors(&query, 2);
            let distances = generate_4bit_dists(&query, &packed, 2);
            for (index, &actual) in distances.iter().enumerate() {
                let low_diff = i32::from(byte & 15) - (index & 15) as i32;
                let high_diff = i32::from(byte >> 4) - (index >> 4) as i32;
                let expected = ((low_diff * low_diff + high_diff * high_diff) as f32).sqrt();
                assert_eq!(actual, expected, "query={byte}, data={index}");
            }
        }
    }

    #[test]
    fn distinct_vectors_produce_distinct_euclidean_distances() {
        let query = to_packed_4bit_vectors(&Array1::from_vec(vec![0, 0, 0]), 3);
        let data = to_packed_4bit_vectors(&Array1::from_vec(vec![0, 0, 0, 3, 4, 0, 15, 0, 0]), 3);
        assert_eq!(generate_4bit_dists(&query, &data, 3), vec![0.0, 5.0, 15.0]);
    }

    #[test]
    fn padding_and_distances_match_scalar_across_vector_boundaries() {
        for dims in [1, 2, 3, 31, 32, 33, 63, 64, 65, 100, 384, 500, 768] {
            let query = Array1::from_iter((0..dims).map(|i| if i % 2 == 0 { 0 } else { 15 }));
            let data = Array1::from_iter(
                query
                    .iter()
                    .copied()
                    .chain(query.iter().map(|&x| 15 - x))
                    .chain((0..dims).map(|i| ((i * 7 + 3) % 16) as u8)),
            );
            let query_wide = to_packed_4bit_vectors(&query, dims);
            let data_wide = to_packed_4bit_vectors(&data, dims);
            let blocks = dims.div_ceil(64);
            assert_eq!(data_wide.len(), 3 * blocks);
            for (packed, expected) in data_wide
                .chunks_exact(blocks)
                .zip(data.as_slice().unwrap().chunks_exact(dims))
            {
                let unpacked: Vec<_> = packed
                    .iter()
                    .flat_map(|block| block.as_array_ref())
                    .flat_map(|&byte| [byte & 15, byte >> 4])
                    .collect();
                assert_eq!(&unpacked[..dims], expected);
                assert!(unpacked[dims..].iter().all(|&x| x == 0));
            }

            let distances = generate_4bit_dists(&query_wide, &data_wide, dims);
            assert_eq!(distances.len(), 3);
            for (&actual, vector) in distances
                .iter()
                .zip(data.as_slice().unwrap().chunks_exact(dims))
            {
                let expected = query
                    .iter()
                    .zip(vector)
                    .map(|(&q, &d)| {
                        let diff = i32::from(q) - i32::from(d);
                        diff * diff
                    })
                    .sum::<i32>() as f32;
                assert_eq!(actual, expected.sqrt(), "dims={dims}");
            }
            assert!(generate_4bit_dists(&query_wide, &[], dims).is_empty());
        }
    }
}
