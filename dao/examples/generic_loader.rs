use bits::{container::Simd256x2, EvpBits};

fn main() {
    let wiki = dao::generic_loader::par_load::<_, half::f16, _, _>(
        "/mnt/bulk/datasets/wikipedia/benchmark-dev-wikipedia-bge-m3-small.h5",
        "train",
        None,
        8192,
        |embedding| embedding.mapv(|f| f.signum()),
    )
    .unwrap();

    dbg!(wiki.len());

    let non_zeros = 200;
    let pubmed = dao::generic_loader::par_load::<_, f32, _, _>(
        "/mnt/bulk/datasets/pubmed/benchmark-dev-pubmed23.h5",
        "train",
        None,
        8192,
        |embedding| EvpBits::<Simd256x2, 384>::from_embedding(embedding, non_zeros),
    )
    .unwrap();

    dbg!(pubmed.len());
}
