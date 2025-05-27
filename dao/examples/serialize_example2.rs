use dao::DaoMetaData;

fn main() -> std::io::Result<()> {
    let f_name = "../_scratch/meta_data.txt";

    let toml = DaoMetaData::from_directory(f_name).unwrap();
    println!("{:?}", toml);

    Ok(())
}
