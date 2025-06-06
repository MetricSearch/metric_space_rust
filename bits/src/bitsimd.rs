use bitvec_simd::BitBlock;
use deepsize::DeepSizeOf;
use std::hash::Hash;
use std::hash::Hasher;
use wide::u64x4;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BitSimd<C, const L: usize> {
    container: C,
}

impl<C, const L: usize> BitSimd<C, L> {
    fn new(container: C) -> Self {
        Self { container }
    }
}

impl<const L: usize> BitSimd<[u64x4; 2], L> {
    // convert total bit to length
    // input: Number of bits
    // output:
    //
    // 1. the number of Vector used
    // 2. after filling 1, the remaining bytes should be filled
    // 3. after filling 2, the remaining bits should be filled
    //
    // notice that this result represents the length of vector
    // so if 3. is 0, it means no extra bits after filling bytes
    // return (length of storage, u64 of last block, bit of last elem)
    // any bits > length of last elem should be set to 0
    #[inline]
    fn bit_to_len(nbits: usize) -> (usize, usize, usize) {
        (
            nbits / (u64x4::BIT_WIDTH as usize),
            (nbits % (u64x4::BIT_WIDTH as usize)) / u64x4::ELEMENT_BIT_WIDTH,
            nbits % u64x4::ELEMENT_BIT_WIDTH,
        )
    }

    pub fn set(&mut self, index: usize, value: bool) {
        if index >= L {
            panic!("cannot set bit {index} in BitSimd of length {L}");
        }

        let (element_index, inner_index, bit_index) = Self::bit_to_len(index);

        let element = &mut self.container[element_index];
        let inner = &mut element.as_array_mut()[inner_index];

        if value {
            *inner |= 1 << bit_index;
        } else {
            *inner &= !(1 << bit_index);
        }
    }

    pub fn and_cloned(&self, other: &Self) -> Self {
        Self {
            container: [
                self.container[0] & other.container[0],
                self.container[1] & other.container[1],
            ],
        }
    }

    pub fn count_ones(&self) -> usize {
        self.container
            .iter()
            .flat_map(|a| a.as_array_ref())
            .map(|e| e.count_ones() as usize)
            .sum()
    }
}

impl<C, const L: usize> DeepSizeOf for BitSimd<C, L> {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        size_of::<C>()
    }
}

impl<const L: usize> Hash for BitSimd<[u64x4; 2], L> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.container
            .iter()
            .flat_map(|e| e.to_array())
            .for_each(|e| e.hash(state));

        // ferdia todo: only hash first L bits
    }
}

#[cfg(test)]
mod tests {
    use crate::bitsimd::BitSimd;
    use wide::u64x4;

    #[test]
    fn bits_default() {
        let bits = BitSimd::<[u64x4; 2], 384>::default();

        assert_eq!(
            &bits
                .container
                .iter()
                .flat_map(|a| a.as_array_ref())
                .copied()
                .collect::<Vec<_>>(),
            &[0, 0, 0, 0, 0, 0, 0, 0]
        );
    }

    #[test]
    fn bits_set_zero() {
        let mut bits = BitSimd::<[u64x4; 2], 384>::default();

        bits.set(0, true);
        assert_eq!(
            &bits
                .container
                .iter()
                .flat_map(|a| a.as_array_ref())
                .copied()
                .collect::<Vec<_>>(),
            &[1, 0, 0, 0, 0, 0, 0, 0]
        );

        bits.set(0, false);
        assert_eq!(bits, BitSimd::default());
    }

    #[test]
    fn bits_all() {
        let mut bits = BitSimd::<[u64x4; 2], 384>::default();

        for i in 0..384 {
            bits.set(i, true);
        }

        assert_eq!(
            &bits
                .container
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
        let mut bits = BitSimd::<[u64x4; 2], 384>::default();

        for i in 0..384 {
            bits.set(i, true);
        }

        bits.set(123, false);

        assert_eq!(
            &bits
                .container
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
        let mut bits = BitSimd::<[u64x4; 2], 384>::default();
        bits.set(65, true);
        assert_eq!(
            &bits
                .container
                .iter()
                .flat_map(|a| a.as_array_ref())
                .copied()
                .collect::<Vec<_>>(),
            &[0, 2, 0, 0, 0, 0, 0, 0]
        );
    }
}
