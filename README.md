# Metric Space Research Kit

## Usage

### Challenge 1

```bash
docker run \
    --pull=always \
    -it \
    --cpus=8 \
    --memory=16g \
    --memory-swap=16g \
    --volume <PATH_TO_HOST_DIR>:<PATH_TO_CONTAINER_DIR>:z \
    ghcr.io/metricsearch/sisap2025:submission \
    /challenge1 <PATH_TO_CONTAINER_DIR>/<INPUT>.h5 <PATH_TO_CONTAINER_DIR>/<OUTPUT>.h5
```

For example

```bash
docker run --pull=always -it --cpus=8 --memory=16g --memory-swap=16g --volume /home/fm208/datasets:/data:z ghcr.io/metricsearch/sisap2025:submission /challenge1 /data/pubmed/benchmark-dev-pubmed23.h5 /data/challenge1_results.h5
```

### Challenge 2

```bash
docker run \
    --pull=always \
    -it \
    --cpus=8 \
    --memory=16g \
    --memory-swap=16g \
    --volume <PATH_TO_HOST_DIR>:<PATH_TO_CONTAINER_DIR>:z \
    ghcr.io/metricsearch/sisap2025:submission \
    /challenge2 <PATH_TO_CONTAINER_DIR>/<INPUT>.h5 <PATH_TO_CONTAINER_DIR>/<OUTPUT>.h5
```

For example

```bash
docker run --pull=always -it --cpus=8 --memory=16g --memory-swap=16g --volume /home/fm208/datasets:/data:z ghcr.io/metricsearch/sisap2025:submission /challenge2 /data/pubmed/benchmark-dev-pubmed23.h5 /data/challenge2_results.h5
```

`podman` required the `:z`, this was not necessary with standard Docker.
