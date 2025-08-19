LAION:

# To create:

cargo run -r --bin create_nn_tables /Volumes/Data/laion/laion-400M-sample/small_set /Users/al/repos/metric_space/_scratch/laion_nn/ 2_200_000 data

Laion:

cargo run -r --bin merge_nn_tables /Users/al/repos/metric_space/_scratch/laion_nn/ /Volumes/Data/laion/laion-400M-sample/small_set /Users/al/repos/metric_space/_scratch/laion_nn/merged data
cargo run -r --bin create_single_table  /Users/al/repos/metric_space/_scratch/laion_nn/ /Users/al/Desktop/merged.bin
cargo run -r --bin cat_nn_bin_table /Users/al/Desktop/merged.bin 1000000

DINO2:

cargo run -r --bin create_nn_tables /Volumes/Data/mf_dino2_h5_batched /Users/al/repos/metric_space/_scratch/mf_dino 200000 embeddings
cargo run -r --bin merge_nn_tables /Users/al/repos/metric_space/_scratch/mf_dino /Volumes/Data/mf_dino2_h5_batched /Users/al/repos/metric_space/_scratch/mf_dino/merged 200000 embeddings
cargo run -r --bin create_single_table  /Users/al/repos/metric_space/_scratch/mf_dino/ /Users/al/Desktop/merged.bin
cargo run -r --bin cat_nn_bin_table /Users/al/repos/metric_space/_scratch/mf_dino/nn_table0.bin 200000

*******

only_merge:

cargo run -r --bin pairwise_merge_nn_tables /Volumes/Data/mf_dino2_h5_batched /Users/al/repos/metric_space/_scratch/mf_dino 200000 embeddings
cargo run -r --bin poly_merge /Users/al/repos/metric_space/_scratch/mf_dino/nn_table /Users/al/repos/metric_space/_scratch/merged
cargo run -r --bin cat_nn_json_table /Users/al/repos/metric_space/_scratch/merged/0.json 10