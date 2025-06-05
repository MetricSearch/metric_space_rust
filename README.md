# Metric Space Research Kit

## Usage

### Cargo

Install the Rust toolchain using `rustup`.

```bash
RUSTFLAGS='-C target-cpu=native' cargo run --release --bin challenge1 -- <PATH_TO_H5_DATASET> <PATH_TO_OUTPUT>
```

and

```bash
RUSTFLAGS='-C target-cpu=native' cargo run --release --bin challenge2 -- <PATH_TO_H5_DATASET> <PATH_TO_OUTPUT>
```

to build and run the `challenge1` and `challenge2` binaries, respectively.

### Docker

```bash
docker run --pull=always -it --cpus=8 --memory=16g --memory-swap=16g --volume <PATH_TO_HOST_DIR>:<PATH_TO_CONTAINER_DIR>:z ghcr.io/metricsearch/sisap2025:latest /challenge1 <PATH_TO_CONTAINER_DIR>/<FILENAME>>.h5
```

For example

```bash
docker run --pull=always -it --cpus=8 --memory=16g --memory-swap=16g --volume /home/fm208/datasets/pubmed:/data:z ghcr.io/metricsearch/sisap2025:latest /challenge1 /data/benchmark-dev-pubmed23.h5
```

`podman` required the `:z`, this was not necessary with standard Docker.
