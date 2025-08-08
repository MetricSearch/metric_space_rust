LAION:

# To create:

cargo run -r --bin create_laion_nn_tables /Volumes/Data/laion/laion-400M-sample/small_set /Users/al/repos/metric_space/_scratch/laion_nn/ 2_200_000 data

# To do the poly phase merge:

cargo run -r --bin merge_laion_nn_tables /Users/al/repos/metric_space/_scratch/laion_nn/ /Volumes/Data/laion/laion-400M-sample/small_set /Users/al/repos/metric_space/_scratch/laion_nn/merged data

# To make a single file:

cargo run -r --bin create_single_table  /Users/al/repos/metric_space/_scratch/laion_nn/ /Users/al/Desktop/merged.bin


# To view:

cargo run -r --bin view_merged /Users/al/Desktop/merged.bin

DINO2:

cargo run -r --bin create_laion_nn_tables /Volumes/Data/mf_dino2_h5_batched /Users/al/repos/metric_space/_scratch/mf_dino 200000 embeddings
cargo run -r --bin merge_laion_nn_tables /Users/al/repos/metric_space/_scratch/mf_dino /Volumes/Data/mf_dino2_h5_batched /Users/al/repos/metric_space/_scratch/mf_dino/merged 200000 embeddings
cargo run -r --bin create_single_table  /Users/al/repos/metric_space/_scratch/mf_dino/ /Users/al/Desktop/merged.bin
cargo run -r --bin view_merged /Users/al/Desktop/merged.bin

*******



