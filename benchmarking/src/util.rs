use std::time::{SystemTime, UNIX_EPOCH};

use rand::{seq::SliceRandom, SeedableRng};
use rand_chacha::ChaCha8Rng;

pub fn perm(size: usize, seed: u64) -> Vec<usize> {
    let mut vec: Vec<usize> = (0..size).collect();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    vec.shuffle(&mut rng);
    vec
}

pub fn perm_system_time(size: usize) -> Vec<usize> {
    perm(
        size,
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    )
}
