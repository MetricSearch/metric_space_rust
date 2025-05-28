use parking_lot::Mutex;
use utils::index::Index;

pub struct Updates {
    inner: Vec<Mutex<Vec<Update>>>,
}

#[derive(Clone, Copy)]
pub struct Update {
    pub index: Index,
    pub sim: f32,
}

impl Updates {
    pub fn new(size: usize) -> Self {
        Self {
            inner: core::iter::repeat_with(|| Mutex::new(vec![]))
                .take(size)
                .collect(),
        }
    } // atomic refer counted pntr

    pub fn add(&self, row: Index, index: Index, sim: f32) {
        self.inner[row.as_usize()]
            .lock()
            .push(Update { index, sim })
    }

    pub fn into_inner(self) -> Vec<Vec<Update>> {
        self.inner
            .into_iter()
            .map(|mutex| mutex.into_inner())
            .collect()
    }

    pub fn len(&self) -> usize {
        self.inner.iter().map(|x| x.lock().len()).sum()
    }
}
