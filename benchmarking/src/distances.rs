use bits::container::{BitsContainer, Simd128, Simd256x2, Simd256x4};
use ndarray::Array1;
use wide::{f32x8, i16x16, i32x8, u16x16, u8x16, u8x32};

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

pub fn to_packed_4bit_vectors(array: &Array1<u8>, dims: usize) -> Vec<u8x32> {
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

pub fn euc_8bit(a: &[u8x32], b: &[u8x32]) -> f32 {
    let mut sum = i32x8::ZERO;

    for (a, b) in a.iter().zip(b.iter()) {
        for (a, b) in a
            .as_array_ref()
            .chunks_exact(16)
            .zip(b.as_array_ref().chunks_exact(16))
        {
            // Need to map into
            let diff = i16x16::from(u8x16::from(a)) - i16x16::from(u8x16::from(b));

            sum += diff.dot(diff);
        }
    }

    (sum.reduce_add() as f32).sqrt()
}

pub fn to_u8x32_vectors(array: &Array1<u8>, dims: usize) -> Vec<u8x32> {
    assert!(dims > 0);
    assert_eq!(array.len() % dims, 0);

    array
        .as_slice()
        .expect("array must be contiguous")
        .chunks_exact(dims)
        .flat_map(|vector| vector.chunks(32).map(u8x32::from))
        .collect()
}

pub fn to_u8_array(array: &Array1<f32>, max_f32: f32) -> Array1<u8> {
    array.mapv(|x| {
        let value = x / max_f32;

        if value.is_nan() {
            // this will never happen
            0
        } else {
            (value * u8::MAX as f32)
                .round()
                .clamp(u8::MIN as f32, u8::MAX as f32) as u8
        }
    })
}

pub fn euc_16bit(a: &[u16x16], b: &[u16x16]) -> f32 {
    let mut sum = f32x8::ZERO;

    for (a, b) in a.iter().zip(b.iter()) {
        for (a, b) in a
            .as_array_ref()
            .chunks_exact(8)
            .zip(b.as_array_ref().chunks_exact(8))
        {
            // Widen before subtraction and squaring; u16 values fit exactly in f32.
            let a = f32x8::new(std::array::from_fn(|lane| f32::from(a[lane])));
            let b = f32x8::new(std::array::from_fn(|lane| f32::from(b[lane])));
            let diff = a - b;
            sum += diff * diff;
        }
    }

    sum.reduce_add().sqrt()
}

pub fn to_u16x16_vectors(array: &Array1<u16>, dims: usize) -> Vec<u16x16> {
    assert!(dims > 0);
    assert_eq!(array.len() % dims, 0);

    array
        .as_slice()
        .expect("array must be contiguous")
        .chunks_exact(dims)
        .flat_map(|vector| vector.chunks(16).map(u16x16::from))
        .collect()
}

pub fn to_u16_array(array: &Array1<f32>, max_f32: f32) -> Array1<u16> {
    array.mapv(|x| {
        let value = x / max_f32;

        if value.is_nan() {
            // this will never happen
            0
        } else {
            (value * u16::MAX as f32)
                .round()
                .clamp(u16::MIN as f32, u16::MAX as f32) as u16
        }
    })
}

pub fn euc_view(a: &[f32x8], b: &[f32x8]) -> f32 {
    let mut sum = f32x8::ZERO;

    for (&a, &b) in a.iter().zip(b.iter()) {
        let diff = a - b;
        sum += diff * diff;
    }

    sum.reduce_add().sqrt()
}

pub fn one_bit_similarity_128(a: &Simd128, b: &Simd128) -> u32 {
    // a.into_u64_iter()
    //     .zip(b.into_u64_iter())
    //     .map(|(x, y)| (x ^ y).count_ones())
    //     .sum::<u32>()

    a.xor(b).count_ones() as u32
}

pub fn one_bit_similarity_256(a: &Simd256x2, b: &Simd256x2) -> u32 {
    // a.into_u64_iter()
    //     .zip(b.into_u64_iter())
    //     .map(|(x, y)| (x ^ y).count_ones())
    //     .sum::<u32>()

    a.xor(b).count_ones() as u32
}

pub fn one_bit_similarity_768(a: &Simd256x4, b: &Simd256x4) -> u32 {
    // a.into_u64_iter()
    //     .zip(b.into_u64_iter())
    //     .map(|(x, y)| (x ^ y).count_ones())
    //     .sum::<u32>()

    a.xor(b).count_ones() as u32
}
