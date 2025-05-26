use std::cmp::Ordering;
use std::fmt::{Display, Formatter};

#[derive(PartialEq, Clone, Copy, Debug)]
pub struct NonNan(f32);

// NonN type for f32 partial comparison
// from https://stackoverflow.com/questions/28247990/how-to-do-a-binary-search-on-a-vec-of-floats

impl NonNan {
    pub fn new(val: f32) -> NonNan {
        if val.is_nan() {
            panic!("Attempted to create a NonNan with NaN value: {val}")
        }

        NonNan(val)
    }

    pub fn as_f32(&self) -> f32 {
        self.0
    }
}

impl Eq for NonNan {}

impl PartialOrd for NonNan {
    fn partial_cmp(&self, other: &NonNan) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Ord for NonNan {
    fn cmp(&self, other: &NonNan) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl Display for NonNan {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<NonNan> for f32 {
    fn from(value: NonNan) -> Self {
        value.as_f32()
    }
}
