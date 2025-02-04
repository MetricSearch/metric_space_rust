use ndarray::{ArrayBase, Ix1, ViewRepr};

pub fn euc(a: ArrayBase<ViewRepr<&f32>,Ix1>, b: ArrayBase<ViewRepr<&f32>,Ix1>) -> f32 {
    f32::sqrt( a.iter().zip(b.iter()).map(|(a, b)| (a - b).powi(2)).sum() )
}

// fast versions for Intel and AARM64 at: https://blog.lancedb.com/my-simd-is-faster-than-yours-fb2989bf25e7/ 