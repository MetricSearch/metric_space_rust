use crate::container::BitsContainer;
use std::arch::x86_64::{__m512i, _mm512_setzero_epi32};

pub type _512 = __m512i;

impl BitsContainer for _512 {
    fn new() -> Self {
        unsafe { _mm512_setzero_epi32() }
    }

    fn count_ones(&self) -> usize {}

    fn and_cloned(&self, other: &Self) -> Self {
        todo!()
    }

    fn set_bit(&mut self, index: usize, value: bool) {
        todo!()
    }

    fn into_u64_iter(&self) -> impl Iterator<Item = u64> {
        todo!()
    }
}

/// Ported from https://stackoverflow.com/questions/50081465/counting-1-bits-population-count-on-large-data-using-avx-512-or-avx-2
/// #   include <immintrin.h>
/// #   include <x86intrin.h>
///
/// uint64_t avx512_vpopcnt(const uint8_t* data, const size_t size) {
///
///     const size_t chunks = size / 64;
///
///     uint8_t* ptr = const_cast<uint8_t*>(data);
///     const uint8_t* end = ptr + size;
///
///     // count using AVX512 registers
///     __m512i accumulator = _mm512_setzero_si512();
///     for (size_t i=0; i < chunks; i++, ptr += 64) {
///
///         // Note: a short chain of dependencies, likely unrolling will be needed.
///         const __m512i v = _mm512_loadu_si512((const __m512i*)ptr);
///         const __m512i p = _mm512_popcnt_epi64(v);
///
///         accumulator = _mm512_add_epi64(accumulator, p);
///     }
///
///     // horizontal sum of a register
///     uint64_t tmp[8] __attribute__((aligned(64)));
///     _mm512_store_si512((__m512i*)tmp, accumulator);
///
///     uint64_t total = 0;
///     for (size_t i=0; i < 8; i++) {
///         total += tmp[i];
///     }
///
///     // popcount the tail
///     while (ptr + 8 < end) {
///         total += _mm_popcnt_u64(*reinterpret_cast<const uint64_t*>(ptr));
///         ptr += 8;
///     }
///
///     while (ptr < end) {
///         total += lookup8bit[*ptr++];
///     }
///
///     return total;
/// }
fn avx512_count_ones() {}

#[cfg(test)]
mod tests {
    use crate::container::{BitsContainer, _512::_512};

    #[test]
    fn bits_default() {
        let bits = <_512>::new();

        assert_eq!(&bits.into_u64_iter().collect::<Vec<_>>(), &[0; 16]);
    }

    #[test]
    fn bits_set_zero() {
        let mut bits = <_512>::new();

        bits.set_bit(0, true);
        assert_eq!(
            &bits.into_u64_iter().collect::<Vec<_>>(),
            &[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        );

        bits.set_bit(0, false);
        assert_eq!(
            bits.into_u64_iter().collect::<Vec<_>>(),
            <_512>::new().into_u64_iter().collect::<Vec<_>>()
        );
    }

    #[test]
    fn bits_all() {
        let mut bits = <_512>::new();

        for i in 0..384 {
            bits.set_bit(i, true);
        }

        assert_eq!(
            &bits.into_u64_iter().collect::<Vec<_>>(),
            &[
                u64::MAX,
                u64::MAX,
                u64::MAX,
                u64::MAX,
                u64::MAX,
                u64::MAX,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        );
    }

    #[test]
    fn bits_all_then_unset() {
        let mut bits = <_512>::new();

        for i in 0..384 {
            bits.set_bit(i, true);
        }

        bits.set_bit(123, false);

        assert_eq!(
            &bits.into_u64_iter().collect::<Vec<_>>(),
            &[
                u64::MAX,
                0b1111011111111111111111111111111111111111111111111111111111111111,
                u64::MAX,
                u64::MAX,
                u64::MAX,
                u64::MAX,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        );
    }

    #[test]
    fn bits_65() {
        let mut bits = <_512>::new();
        bits.set_bit(65, true);
        assert_eq!(
            &bits.into_u64_iter().collect::<Vec<_>>(),
            &[0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]
        );
    }
}
