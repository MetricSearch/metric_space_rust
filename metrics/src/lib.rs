use ndarray::{Array, Array1, ArrayBase, Ix1, ViewRepr};

pub fn euc(a: ArrayBase<ViewRepr<&f32>,Ix1>, b: ArrayBase<ViewRepr<&f32>,Ix1>) -> f32 {
    f32::sqrt( a.iter().zip(b.iter()).map(|(a, b)| (a - b).powf(2.0)).sum() )
}