FROM lukemathwalker/cargo-chef:latest-rust-1 AS chef
WORKDIR /app

# prepare toolchain
RUN rustup target add x86_64-unknown-linux-gnu

# add dependencies
RUN apt-get update && apt-get install cmake -y

# prepare recipe
FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# create a new layer using only the `recipe.json` as input (so cached if it does not change)
FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json

# E5-2690 v4
ENV RUSTFLAGS='-C target-cpu=broadwell'
ARG RUST_ARGS='--release --package sisap2025 --target x86_64-unknown-linux-gnu'

# build dependencies
RUN cargo chef cook $RUST_ARGS --recipe-path recipe.json

# build app
COPY . .
RUN touch Cargo.toml
RUN cargo build $RUST_ARGS

FROM ubuntu
COPY --from=builder /app/target/x86_64-unknown-linux-gnu/release/challenge2 .
COPY --from=builder /app/target/x86_64-unknown-linux-gnu/release/challenge1_rev .
COPY --from=builder /app/target/x86_64-unknown-linux-gnu/release/challenge2_dino2 .
