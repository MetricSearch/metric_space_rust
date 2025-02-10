use dao::{DaoMetaData, Normed};
use serde_json;
use std::fs::File;
use std::io::{Read, Write};

fn main() -> std::io::Result<()> {
    let dao_meta = DaoMetaData {
        name: "Test".to_string(),
        description: "This is test".to_string(),
        data_disk_format: "csv_f32".to_string(),
        path_to_data: "/Data/somewhere.csv".to_string(),
        normed: Normed::L2,
        num_records: 100,
        dim: 384,
    };

    let f_name = "../_scratch/meta_data.txt";

    let toml = toml::to_string(&dao_meta).unwrap();

    let mut file = File::create(f_name)?;
    file.write(toml.as_ref())?;

    let mut file = File::open(f_name)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    let toml2: DaoMetaData = toml::from_str(&contents).unwrap();
    println!("{:?}", toml2);

    Ok(())
}
