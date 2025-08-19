LAION:

RUSTFLAGS=-Awarnings cargo build -r --all-targets

 files:
/Users/al/repos/metric_space/_scratch/laion_nn/
/Users/al/Desktop/merged.bin

DINO2:

cargo run -r --bin create_nn_tables /Volumes/Data/mf_dino2_h5_batched /Users/al/repos/metric_space/_scratch/mf_dino 200000 embeddings
cargo run -r --bin pairwise_merge_nn_tables /Volumes/Data/mf_dino2_h5_batched /Users/al/repos/metric_space/_scratch/mf_dino 200000 embeddings
cargo run -r --bin poly_merge /Users/al/repos/metric_space/_scratch/mf_dino/nn_table /Users/al/repos/metric_space/_scratch/merged
cargo run -r --bin cat_nn_json_table /Users/al/repos/metric_space/_scratch/merged/0.json 10