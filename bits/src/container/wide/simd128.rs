use crate::container::BitsContainer;
use wide::u64x2;

pub type Simd128 = u64x2;

impl BitsContainer for Simd128 {
    fn new() -> Self {
        Self::default()
    }

    fn count_ones(&self) -> usize {
        self.as_array_ref()
            .iter()
            .map(|e| e.count_ones() as usize)
            .sum()
    }

    fn and_cloned(&self, other: &Self) -> Self {
        *self & *other
    }

    fn xor(&self, other: &Self) -> Self {
        *self ^ *other
    }

    fn set_bit(&mut self, index: usize, value: bool) {
        let element_width = size_of::<u64x2>() * 8;
        let inner_width = size_of::<u64>() * 8;

        if index >= element_width {
            panic!()
        }

        let inner_index = (index % element_width) / inner_width;
        let bit_index = index % inner_width;

        let inner = &mut self.as_array_mut()[inner_index];

        if value {
            *inner |= 1 << bit_index;
        } else {
            *inner &= !(1 << bit_index);
        }
    }

    fn into_u64_iter(&self) -> impl Iterator<Item = u64> {
        self.to_array().into_iter()
    }
}

#[cfg(test)]
mod tests {
    use crate::container::{BitsContainer, Simd128};

    #[test]
    fn bits_default() {
        let bits = <Simd128>::default();

        assert_eq!(&bits.into_u64_iter().collect::<Vec<_>>(), &[0; 2]);
    }

    #[test]
    fn bits_set_zero() {
        let mut bits = <Simd128>::default();

        bits.set_bit(0, true);
        assert_eq!(&bits.into_u64_iter().collect::<Vec<_>>(), &[1, 0]);

        bits.set_bit(0, false);
        assert_eq!(bits, <Simd128>::default());
    }

    #[test]
    fn bits_all() {
        let mut bits = <Simd128>::default();

        for i in 0..128 {
            bits.set_bit(i, true);
        }

        assert_eq!(
            &bits.into_u64_iter().collect::<Vec<_>>(),
            &[u64::MAX, u64::MAX,]
        );
    }

    #[test]
    fn bits_all_then_unset() {
        let mut bits = <Simd128>::default();

        for i in 0..128 {
            bits.set_bit(i, true);
        }

        bits.set_bit(123, false);

        assert_eq!(
            &bits.into_u64_iter().collect::<Vec<_>>(),
            &[
                u64::MAX,
                0b1111011111111111111111111111111111111111111111111111111111111111,
            ]
        );
    }

    #[test]
    fn bits_65() {
        let mut bits = <Simd128>::default();
        bits.set_bit(65, true);
        assert_eq!(&bits.into_u64_iter().collect::<Vec<_>>(), &[0, 2]);
    }
}
