// Converts vectors of distances into vectors of indices and distances
pub fn arg_sort_2d<T: PartialOrd + Copy>(dists: Vec<Vec<T>>) -> (Vec<Vec<usize>>, Vec<Vec<T>>) {
    dists
        .iter()
        .map(|vec| {
            let mut enumerated = vec.iter().enumerate().collect::<Vec<(usize, &T)>>();

            enumerated.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());

            enumerated.into_iter().unzip()
        })
        .collect()
}

// Converts vectors of distances into vectors of indices and distances
pub fn arg_sort<T: PartialOrd + Copy>(dists: Vec<T>) -> (Vec<usize>, Vec<T>) {
    let mut enumerated = dists.iter().enumerate().collect::<Vec<(usize, &T)>>();

    enumerated.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());

    enumerated.into_iter().unzip()
}
