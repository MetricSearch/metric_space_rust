use anyhow::Result;
use dao::glove100_hdf5_dao_loader::hdf5_glove_f32_load;

fn main() -> Result<()> {

    let f_name = "/Volumes/Data/glove-100-angular.hdf5";

    let dao = hdf5_glove_f32_load( f_name ).unwrap();

    println!( "found {:?}", dao.embeddings.shape() );


    println!("Read from file: {:?}", f_name);

    Ok(())
}
