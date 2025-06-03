use anyhow::Result;
use dao::hdf5_to_dao_loader::hdf5_f32_to_bsp_load;

fn main() -> Result<()> {
    let f_name = "/Volumes/Data/sisap_challenge_25/pubmed/benchmark-dev-pubmed23.h5";
    let dao = hdf5_f32_to_bsp_load(f_name, 200, 10, 200).unwrap();

    println!("Read from file: {:?}", f_name);
    println!("Loaded {:?}", dao.embeddings.shape());

    Ok(())
}
