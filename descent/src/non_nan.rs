use std::cmp::Ordering;
use std::fmt::{Display, Formatter};

#[derive(PartialEq, PartialOrd, Clone, Copy, Debug)]
pub struct NonNan(pub f32);

// NonN type for f32 partial comparison
// from https://stackoverflow.com/questions/28247990/how-to-do-a-binary-search-on-a-vec-of-floats

impl NonNan {
    pub(crate) fn new(val: f32) -> Option<NonNan> {
        if val.is_nan() {
            None
        } else {
            Some(NonNan(val))
        }
    }
}

impl Eq for NonNan {}

impl Ord for NonNan {
    fn cmp(&self, other: &NonNan) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl Display for NonNan {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0 )
    }
}
