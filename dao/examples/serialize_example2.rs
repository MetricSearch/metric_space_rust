use std::fs::File;
use std::io::{Read, Write};
use serde_json;
use dao::{dao_metadata_from_dir, DaoMetaData, Normed};

fn main() -> std::io::Result<()> {

    let f_name = "../_scratch/meta_data.txt";

    let toml: DaoMetaData = dao_metadata_from_dir(f_name).unwrap();
    println!("{:?}", toml);

    Ok(())
}
