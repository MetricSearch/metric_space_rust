#!/bin/bash
cargo bench --bench multiple_one_bit
cargo bench --bench multiple_eucs
cargo bench --bench evp_100dim
cargo bench --bench evp_384dim
cargo bench --bench evp_500dim
cargo bench --bench evp_768dim
cargo bench --bench multiple_bit_eucs