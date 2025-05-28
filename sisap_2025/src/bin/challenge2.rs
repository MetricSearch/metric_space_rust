/*
In this task, participants are asked to develop memory-efficient indexing solutions that will be used to compute an approximation of the k-nearest neighbor graph for k=15. Each solution will be run in a Linux container with limited memory and storage resources.
Container specifications: 8 virtual CPUs, 16 GB of RAM, the dataset will be mounted read-only into the container.
Wall clock time for the entire task: 12 hours.
Minimum average recall to be considered in the final ranking: 0.8.
Dataset: GOOAQ (3 million vectors (384 dimensions) ).
The goal is to compute the k-nearest neighbor graph (without self-references), i.e., find the k-nearest neighbors using all objects in the dataset as queries.
We will measure graph’s quality as the recall against a provided gold standard and the full computation time (i.e., including preprocessing, indexing, and search, and postprocessing)
We provide a development dataset; the evaluation phase will use an undisclosed dataset of similar size computed with the same neural model.
*/

fn main() {
    println!("hello world!")
}
