use std::fs::File;
use std::io::{Read, Write};
use serde_json;
use dao::{get_dao_metadata, DaoMetaData, Normed};

fn main() -> std::io::Result<()> {

    let f_name = "../_scratch/meta_data.txt";

    let toml: DaoMetaData = get_dao_metadata(f_name).unwrap();
    println!("{:?}", toml);

    Ok(())
}
