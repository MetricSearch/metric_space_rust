use std::sync::{Arc, Mutex};

pub  struct Updates {
    inner: Arc<Mutex<Vec<Vec<Update>>>>,
}

#[derive (Clone,Copy)]
pub struct Update{ pub index: usize, pub sim: f32 }

impl Updates {
    pub fn new(size: usize) -> Self {
        Self{ inner: Arc::new( Mutex::new(vec![vec![];size] ) ) }
    } // atomic refer counted pntr

    pub fn add(&self, row: usize, index: usize, sim: f32) {
        self.inner.lock().unwrap()[row].push(Update{index,sim})
    }

    pub fn into_inner(self) -> Vec<Vec<Update>> {
        Arc::into_inner(self.inner).unwrap().into_inner().unwrap()
    }


    // if ! neighbours.row(u1_id).iter().any(|x| *x == u2_id) { // Matlab line 193
    //     let position = index_of_min(&similarities.row(u1_id));
    //     neighbours[[u1_id,position]] = u2_id;
    //     similarities[[u1_id,position]] = this_sim;
    //     neighbour_is_new[[u1_id,position]] = true;
    //     global_mins[u1_id] = minimum_in(&similarities.row(u1_id));  // Matlab line 198
    //     work_done = work_done + 1;
    // }
}
