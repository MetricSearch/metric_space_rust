use deepsize::DeepSizeOf;
use hdf5::H5Type;
use ndarray::{Dim, Dimension, NdIndex, SliceInfoElem, SliceNextDim};
use serde::{Deserialize, Serialize};
use std::{
    fmt::{self, Display},
    ops::{Add, AddAssign, Mul},
};

/// Index into an array
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, Hash, DeepSizeOf)]
pub struct Index(u32);

impl Index {
    pub const MAX: Self = Self(u32::MAX);
    pub const ZERO: Self = Self(0);

    pub fn new(index: u32) -> Self {
        Self(index)
    }

    pub fn from_usize(index: usize) -> Self {
        Self(u32::try_from(index).unwrap())
    }

    pub fn as_usize(&self) -> usize {
        self.0.try_into().unwrap()
    }
}

impl Display for Index {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<usize> for Index {
    fn from(value: usize) -> Self {
        Self::from_usize(value)
    }
}

impl From<Index> for usize {
    fn from(value: Index) -> Self {
        value.as_usize()
    }
}

impl Add for Index {
    type Output = Index;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.0 + rhs.0)
    }
}

impl Mul for Index {
    type Output = Index;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(self.0 * rhs.0)
    }
}

impl Add<u32> for Index {
    type Output = Index;

    fn add(self, rhs: u32) -> Self::Output {
        Self::new(self.0 + rhs)
    }
}

impl AddAssign<u32> for Index {
    fn add_assign(&mut self, rhs: u32) {
        self.0 += rhs;
    }
}

impl num_traits::identities::Zero for Index {
    fn zero() -> Self {
        Self::ZERO
    }

    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl num_traits::identities::One for Index {
    fn one() -> Self {
        Self::new(1)
    }
}

// unsafe impl NdIndex<Dim<[usize; 1]>> for [Index; 1] {
//     fn index_checked(&self, dim: &D, strides: &D) -> Option<isize> {
//         // todo!self.as_usize()
//         todo!()
//     }

//     fn index_unchecked(&self, strides: &D) -> isize {
//         todo!()
//     }
// }
// unsafe impl NdIndex<Dim<[usize; 1]>> for [Index; 2] {
//     fn index_checked(&self, dim: &D, strides: &D) -> Option<isize> {
//         // todo!self.as_usize()
//         todo!()
//     }

//     fn index_unchecked(&self, strides: &D) -> isize {
//         todo!()
//     }
// }

impl SliceNextDim for Index {
    type InDim = <usize as SliceNextDim>::InDim;

    type OutDim = <usize as SliceNextDim>::OutDim;
}

impl From<Index> for SliceInfoElem {
    fn from(value: Index) -> Self {
        value.as_usize().into()
    }
}

unsafe impl H5Type for Index {
    fn type_descriptor() -> hdf5::types::TypeDescriptor {
        <usize as H5Type>::type_descriptor()
    }
}
