use crate::{KnnSearch, RDescent};
use dao::Dao;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};
use std::rc::Rc;
use utils::non_nan::NonNan;
use utils::pair::Pair;

// todo: FERDIA SAYS THIS SHOULD BE HASH NOT HASHER, LET YE BE WARNED
impl<T: Clone + Default + Hasher> KnnSearch<T> for RDescent {
    fn knn_search(
        &self,
        query: T,
        dao: Rc<Dao<T>>,
        num_neighbours: usize,
        distance: fn(&T, &T) -> f32,
    ) -> (usize, Vec<Pair>) {
        let mut visited_set: HashSet<usize, BuildHasherDefault<T>> = HashSet::default();
        let entry_point = 0; // <<<<<<<<<<<<<< TODO ENTRY POINT OF ZERO FOR NOW
        let ep_q_dist = NonNan::new(distance(&query, dao.get_datum(0)));
        let mut results_list: BinaryHeap<Pair> = BinaryHeap::new(); // biggest first - a max-heap
        let mut candidates_list: BinaryHeap<Reverse<Pair>> = BinaryHeap::new(); // in reverse order - smallest first
        candidates_list.push(Reverse(Pair::new(ep_q_dist, entry_point)));
        visited_set.insert(entry_point);

        let mut finished = false;

        while !finished {
            if candidates_list.len() == 0 {
                finished = true;
            } else {
                let nearest_candidate_pair = candidates_list.pop().unwrap().0; // 0 to take Reverse  - get the nearest element to q (sort order for C) - extract the reverse

                if !results_list.is_empty()
                    && nearest_candidate_pair.distance > results_list.peek().unwrap().distance
                {
                    // if furthest than the furthest we are done
                    finished = true;
                } else {
                    results_list.push(Pair::new(
                        nearest_candidate_pair.distance,
                        nearest_candidate_pair.index,
                    )); // should be a copy perhaps?
                    if results_list.len() > num_neighbours {
                        // was ef <<<<<<
                        // might not be full so check length after push
                        results_list.pop();
                    }
                    let neighbours_of_nearest_candidate =
                        &self.neighbours.row(nearest_candidate_pair.index); // List<Integer> - nns of nearest_candidate

                    let new_cands: Vec<Reverse<Pair>> = neighbours_of_nearest_candidate
                        .into_iter()
                        .map(|&x| (x, dao.get_datum(x)))
                        .into_iter()
                        .filter_map(|neighbour_index| {
                            if visited_set.contains(&neighbour_index.0) {
                                None
                            } else {
                                visited_set.insert(neighbour_index.0);
                                Some(neighbour_index)
                            }
                        })
                        .map(|unseen_neighbour| {
                            let distance_q_next_neighbour =
                                NonNan::new(distance(&query, &unseen_neighbour.1));

                            // let distance_q_next_neighbour = dist_fn(&query, &unseen_neighbour.1);
                            Reverse(Pair::new(distance_q_next_neighbour, unseen_neighbour.0))
                        })
                        .collect();

                    candidates_list.extend(new_cands);
                }
            }
        }

        return (candidates_list.len(), results_list.into_sorted_vec()); /* distances plus Vec<Pair> */
    }
}
