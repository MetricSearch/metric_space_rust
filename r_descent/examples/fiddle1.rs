use ndarray::Array;
use ndarray::{ArrayBase, Dim, OwnedRepr};

fn main() {
    let mut array: ArrayBase<OwnedRepr<usize>, Dim<[usize; 2]>> = Array::zeros((3, 4));

    let _: &usize = array.get((2, 3)).unwrap();

    let a_loc: &mut usize = array.get_mut((1, 3)).unwrap();
    *a_loc = 5;

    println!("My array {:?}!", array);
}
