//! Implementation of the Heap for Descent.
//! The heap contains 3 data structures - all 2D Matrices
//! All the matrices are the same shape - number of items in the data set X num near neighbours
//! Each row represents a data point in the dats set
//! the dta structures are:
//! indices - which contain the indices of the near neighbours of the
//! distances - the distances from the node corresponding to the row to each of the near neighbours
//! flags - misc flags used to record meta information about the entries WHAT ARE THEY??

//use ndarray::{indices, Array, ArrayBase, Dim, OwnedRepr, ViewRepr};
//use crate::descent::NonNan;

pub struct Heap {
    pub num_entries : usize,
    pub num_neighbours : usize,
    pub indices : Vec<Vec<i32>>, // ArrayBase<OwnedRepr<i32>, Dim<[usize; 2]>>,
    pub distances : Vec<Vec<f32>>, // ArrayBase<OwnedRepr<NonNan>, Dim<[usize; 2]>>,
    pub flags : Vec<Vec<u8>>,  //ArrayBase<OwnedRepr<u8>, Dim<[usize; 2]>>,
    //maybe bitset?
}

impl Heap {
    pub fn new(num_entries: usize, num_neighbours: usize) -> Heap {
        let indices = vec![vec![-1; num_neighbours]; num_entries]; // Array::from_elem((num_entries, num_neighbours), -1);
        let distances =  vec![vec![f32::MAX; num_neighbours]; num_entries];  // Array::from_elem((num_entries, num_neighbours), NonNan::new(f32::MAX).unwrap());
        let flags = vec![vec![0; num_neighbours]; num_entries]; // Array::zeros((num_entries, num_neighbours));

        Self {num_entries, num_neighbours, indices, distances, flags}
    }
}