use utils::index::Index;
use utils::ndcg;
use utils::non_nan::NonNan;
use utils::pair::Pair;

pub fn main() {
    let mut perfect_results: Vec<Pair> = Vec::new();
    perfect_results.push(Pair::new(NonNan::new(1.0), Index::new(5)));
    perfect_results.push(Pair::new(NonNan::new(1.2), Index::new(9)));
    perfect_results.push(Pair::new(NonNan::new(1.5), Index::new(7)));
    perfect_results.push(Pair::new(NonNan::new(2.0), Index::new(6)));

    let mut bad_results: Vec<Pair> = Vec::new();
    bad_results.push(Pair::new(NonNan::new(1.0), Index::new(10)));
    bad_results.push(Pair::new(NonNan::new(1.2), Index::new(20)));
    bad_results.push(Pair::new(NonNan::new(1.5), Index::new(30)));
    bad_results.push(Pair::new(NonNan::new(2.0), Index::new(40)));

    let mut mixed_results: Vec<Pair> = Vec::new();
    mixed_results.push(Pair::new(NonNan::new(1.0), Index::new(5)));
    mixed_results.push(Pair::new(NonNan::new(1.2), Index::new(9)));
    mixed_results.push(Pair::new(NonNan::new(1.5), Index::new(3)));
    mixed_results.push(Pair::new(NonNan::new(2.0), Index::new(4)));

    let mut gt: Vec<Pair> = Vec::new();
    gt.push(Pair::new(NonNan::new(1.2), Index::new(5)));
    gt.push(Pair::new(NonNan::new(1.5), Index::new(9)));
    gt.push(Pair::new(NonNan::new(1.7), Index::new(7)));
    gt.push(Pair::new(NonNan::new(2.0), Index::new(6)));

    println!("Gt and gt = {} should be 1", ndcg(&gt, &gt)); // Should be 1.
    println!(
        "perfect and gt = {} should be 1",
        ndcg(&perfect_results, &gt)
    ); // Should be 1
    println!("bad and gt = {} should be 0", ndcg(&bad_results, &gt)); // Should be 0
    println!(
        "mixed and gt = {} should be between 0 and 1",
        ndcg(&mixed_results, &gt)
    ); // Should be somewhere between 0 and 1.
}
