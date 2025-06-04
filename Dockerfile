FROM lukemathwalker/cargo-chef:latest-rust-1 AS chef
WORKDIR /app

# prepare toolchain
RUN rustup target add x86_64-unknown-linux-musl

# add musl tools
RUN apt-get update && apt-get install musl-tools clang llvm cmake -y

# prepare recipe
FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# create a new layer using only the `recipe.json` as input (so cached if it does not change)
FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json

ENV RUSTFLAGS='-C target-cpu=native'
ARG RUST_ARGS='--release --all-targets --target x86_64-unknown-linux-musl'

# build dependencies
RUN cargo chef cook $RUST_ARGS --recipe-path recipe.json

# build app
COPY . .
RUN touch Cargo.toml
RUN cargo build $RUST_ARGS

FROM scratch
COPY --from=builder /app/target/x86_64-unknown-linux-musl/release/challenge1 .
COPY --from=builder /app/target/x86_64-unknown-linux-musl/release/challenge2 .
