use crate::container::BitsContainer;
use wide::u64x4;

pub type Simd256x4 = [u64x4; 4];

impl BitsContainer for Simd256x4 {
    fn new() -> Self {
        Self::default()
    }

    fn count_ones(&self) -> usize {
        self.iter()
            .flat_map(|a| a.as_array_ref())
            .map(|e| e.count_ones() as usize)
            .sum()
    }

    fn and_cloned(&self, other: &Self) -> Self {
        [
            self[0] & other[0],
            self[1] & other[1],
            self[2] & other[2],
            self[3] & other[3],
        ]
    }

    fn set_bit(&mut self, index: usize, value: bool) {
        let element_width = size_of::<u64x4>() * 8;
        let inner_width = size_of::<u64>() * 8;

        let element_index = index / element_width;
        let inner_index = (index % element_width) / inner_width;
        let bit_index = index % inner_width;

        let element = &mut self[element_index];
        let inner = &mut element.as_array_mut()[inner_index];

        if value {
            *inner |= 1 << bit_index;
        } else {
            *inner &= !(1 << bit_index);
        }
    }

    fn into_u64_iter(&self) -> impl Iterator<Item = u64> {
        self.iter().flat_map(|e| e.to_array())
    }
}

#[cfg(test)]
mod tests {
    use crate::container::{BitsContainer, Simd256x4};

    #[test]
    fn bits_default() {
        let bits = <Simd256x4>::new();

        assert_eq!(&bits.into_u64_iter().collect::<Vec<_>>(), &[0; 16]);
    }

    #[test]
    fn bits_set_zero() {
        let mut bits = <Simd256x4>::new();

        bits.set_bit(0, true);
        assert_eq!(
            &bits.into_u64_iter().collect::<Vec<_>>(),
            &[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        );

        bits.set_bit(0, false);
        assert_eq!(bits, <Simd256x4>::new());
    }

    #[test]
    fn bits_all() {
        let mut bits = <Simd256x4>::new();

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
        let mut bits = <Simd256x4>::new();

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
        let mut bits = <Simd256x4>::new();
        bits.set_bit(65, true);
        assert_eq!(
            &bits.into_u64_iter().collect::<Vec<_>>(),
            &[0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]
        );
    }
}
