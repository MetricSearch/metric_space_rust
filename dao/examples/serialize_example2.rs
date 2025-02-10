use dao::{dao_metadata_from_dir};
use serde_json;

fn main() -> std::io::Result<()> {
    let f_name = "../_scratch/meta_data.txt";

    let toml = dao_metadata_from_dir(f_name).unwrap();
    println!("{:?}", toml);

    Ok(())
}
