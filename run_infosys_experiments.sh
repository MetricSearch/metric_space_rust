#!/bin/bash

OUT_DIR = "infosys_raw_data"

# Make the output dir if not there
mkdir -p $OUT_DIR

for LOOP in {1..10}
do
    {
    cargo run -r --example evp_100dim_brute_force_example 
    cargo run -r --example evp_384dim_brute_force_example
    cargo run -r --example evp_500dim_brute_force_example
    cargo run -r --example evp_768dim_brute_force_example 

    cargo run -r --example 4bit_brute_force_example      
    cargo run -r --example 8bit_brute_force_example
    cargo run -r --example 16bit_brute_force_example         

    cargo run -r --example f32_euc_brute_force_example

    cargo run -r --example hamming_100dim_brute_force_example
    cargo run -r --example hamming_384dim_brute_force_example
    cargo run -r --example hamming_500dim_brute_force_example
    cargo run -r --example hamming_768dim_brute_force_example
    } > $OUT_DIR/$LOOP.txt
done