use std::cmp::Ordering;
use std::fmt::{Display, Formatter};

#[derive(PartialEq, Clone, Copy, Debug)]
pub struct NonNanF64(f64);

// NonN type for f32 partial comparison
// from https://stackoverflow.com/questions/28247990/how-to-do-a-binary-search-on-a-vec-of-floats

impl NonNanF64 {
    pub fn new(val: f64) -> NonNanF64 {
        if val.is_nan() {
            panic!("Attempted to create a NonNan64 with NaN value: {val}")
        }

        NonNanF64(val)
    }

    pub fn as_f64(&self) -> f64 {
        self.0
    }
}

impl Eq for NonNanF64 {}

impl PartialOrd for NonNanF64 {
    fn partial_cmp(&self, other: &NonNanF64) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Ord for NonNanF64 {
    fn cmp(&self, other: &NonNanF64) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl Display for NonNanF64 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<NonNanF64> for f64 {
    fn from(value: NonNanF64) -> Self {
        value.as_f64()
    }
}
