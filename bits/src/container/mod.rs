use std::hash::Hasher;

mod wide;

pub trait BitsContainer {
    fn count_ones(&self) -> usize;
    fn and_cloned(&self, other: &Self) -> Self;
    fn set_bit(&mut self, index: usize, value: bool);

    /// Hash with width (W) limit
    // also avoids "cannot impl foreign trait (Hash) for foreign type" issue
    fn hash<H: Hasher, const W: usize>(&self, state: &mut H);
}

#[cfg(test)]
mod tests {
    use crate::container::BitsContainer;
    use wide::u64x4;

    #[test]
    fn bits_default() {
        let bits = <[u64x4; 2]>::default();

        assert_eq!(
            &bits
                .iter()
                .flat_map(|a| a.as_array_ref())
                .copied()
                .collect::<Vec<_>>(),
            &[0, 0, 0, 0, 0, 0, 0, 0]
        );
    }

    #[test]
    fn bits_set_zero() {
        let mut bits = <[u64x4; 2]>::default();

        bits.set_bit(0, true);
        assert_eq!(
            &bits
                .iter()
                .flat_map(|a| a.as_array_ref())
                .copied()
                .collect::<Vec<_>>(),
            &[1, 0, 0, 0, 0, 0, 0, 0]
        );

        bits.set_bit(0, false);
        assert_eq!(bits, <[u64x4; 2]>::default());
    }

    #[test]
    fn bits_all() {
        let mut bits = <[u64x4; 2]>::default();

        for i in 0..384 {
            bits.set_bit(i, true);
        }

        assert_eq!(
            &bits
                .iter()
                .flat_map(|a| a.as_array_ref())
                .copied()
                .collect::<Vec<_>>(),
            &[
                u64::MAX,
                u64::MAX,
                u64::MAX,
                u64::MAX,
                u64::MAX,
                u64::MAX,
                0,
                0
            ]
        );
    }

    #[test]
    fn bits_all_then_unset() {
        let mut bits = <[u64x4; 2]>::default();

        for i in 0..384 {
            bits.set_bit(i, true);
        }

        bits.set_bit(123, false);

        assert_eq!(
            &bits
                .iter()
                .flat_map(|a| a.as_array_ref())
                .copied()
                .collect::<Vec<_>>(),
            &[
                u64::MAX,
                0b1111011111111111111111111111111111111111111111111111111111111111,
                u64::MAX,
                u64::MAX,
                u64::MAX,
                u64::MAX,
                0,
                0
            ]
        );
    }

    #[test]
    fn bits_65() {
        let mut bits = <[u64x4; 2]>::default();
        bits.set_bit(65, true);
        assert_eq!(
            &bits
                .iter()
                .flat_map(|a| a.as_array_ref())
                .copied()
                .collect::<Vec<_>>(),
            &[0, 2, 0, 0, 0, 0, 0, 0]
        );
    }
}
