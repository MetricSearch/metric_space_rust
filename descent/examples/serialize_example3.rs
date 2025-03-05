use ndarray::Array;
use serde_json;

fn main() {
    let mut array = Array::zeros((3, 4));

    let _: &usize = array.get((2, 3)).unwrap();

    let a_loc: &mut usize = array.get_mut((1, 3)).unwrap();
    *a_loc = 5;

    let serialised = serde_json::to_string(&array);
    println!("serialised: {:?}", serialised);
}
