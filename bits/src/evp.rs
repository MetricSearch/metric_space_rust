use crate::container::BitsContainer;
use deepsize::DeepSizeOf;
use std::hash::Hash;
use std::hash::Hasher;

/// Bit Scalar Product using bit container C, with actual width W
///
/// For example, an AVX512 container used to store 384 bits.
#[derive(Debug, Clone, Default)]
pub struct EvpBits<C, const W: usize> {
    ones: C,
    negative_ones: C,
}

impl<C: BitsContainer, const W: usize> EvpBits<C, W> {
    pub fn new(ones: C, negative_ones: C) -> Self {
        Self {
            ones,
            negative_ones,
        }
    }
}

impl<C, const W: usize> DeepSizeOf for EvpBits<C, W> {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        size_of::<C>() * 2
    }
}

impl<C: BitsContainer, const W: usize> Hash for EvpBits<C, W> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.ones.hash::<_, W>(state);
        self.negative_ones.hash::<_, W>(state);
    }
}

#[inline(always)]
pub fn bsp_distance<C: BitsContainer, const W: usize>(
    a: &EvpBits<C, W>,
    b: &EvpBits<C, W>,
) -> usize {
    let aa = a.ones.and_cloned(&b.ones).count_ones();
    let bb = a.negative_ones.and_cloned(&b.negative_ones).count_ones();
    let cc = a.ones.and_cloned(&b.negative_ones).count_ones();
    let dd = b.ones.and_cloned(&a.negative_ones).count_ones();

    (cc + dd + W * 2) - (aa + bb)
}

#[inline(always)]
pub fn bsp_similarity<C: BitsContainer, const W: usize>(
    a: &EvpBits<C, W>,
    b: &EvpBits<C, W>,
) -> usize {
    let aa = a.ones.and_cloned(&b.ones).count_ones();
    let bb = a.negative_ones.and_cloned(&b.negative_ones).count_ones();
    let cc = a.ones.and_cloned(&b.negative_ones).count_ones();
    let dd = b.ones.and_cloned(&a.negative_ones).count_ones();

    // println!( "a {:?} b {:?}", a, b) ;

    // adding X * 256 * 2 means the result must be positive since second term is maximally X * 256 * 2  ( BitVecSimd<[u64x4; X],4> )
    // min is zero (not in practice) max is 2048 (not in practice).
    // println!("A={} B={} C={} D={} result ={}", A, B, C, D, (A + B+X*256*2) - (C + D ));

    (aa + bb + W * 2) - (cc + dd)
}
