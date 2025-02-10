use ndarray::{Array, Array2};

fn main() -> std::io::Result<()> {
    let data_vec: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    // The object that we will serialize.
    let arrai = Array::from_shape_vec((3, 3), data_vec);

    println!("{:?}", &arrai);

    let arrai = arrai.unwrap();

    let encoded = bincode::serialize(&arrai).unwrap();

    let decoded: Array2<f32> = bincode::deserialize(&encoded).unwrap();

    println!("Decoded: {:?}", decoded);
    Ok(())
}
