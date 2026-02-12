#!/bin/bash
cargo bench --bench multiple_one_bit
cargo bench --bench multiple_bit_eucs
cargo bench --bench pca
cargo bench --bench scalar_product_distance
cargo bench --bench evp_different_x
cargo bench --bench evp_100dim
cargo bench --bench evp_384dim
cargo bench --bench evp_500dim
cargo bench --bench evp_768dim