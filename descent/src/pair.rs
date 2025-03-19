use std::cmp::Ordering;
use crate::non_nan::NonNan;

pub struct Pair{
    pub distance: NonNan,
    pub index: usize,
}

impl Pair {
    pub(crate) fn new(distance: NonNan, index: usize) -> Pair {
        Pair { distance, index }
    }
}

impl Eq for Pair { // Marker Trait
}

impl PartialEq for Pair {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl PartialOrd for Pair {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl Ord for Pair {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(&other).unwrap()
    }
}