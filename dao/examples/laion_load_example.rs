use anyhow::Result;
use dao::laion_10M_hdf5_dao_loader::hdf5_laion_f32_load;

fn main() -> Result<()> {

    let f_name = "/Volumes/Data/laion/laion2B-en-clip768v2-n=10M.h5";

    let dao = hdf5_laion_f32_load( f_name ).unwrap();

    println!( "found {:?}", dao.embeddings.shape() );


    println!("Read from file: {:?}", f_name);

    Ok(())
}
