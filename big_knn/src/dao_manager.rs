use bits::container::BitsContainer;
use bits::EvpBits;
use dao::Dao;
use std::rc::Rc;
use utils::address::{GlobalAddress, LocalAddress};

pub struct DaoStore<C: BitsContainer, const W: usize> {
    daos: Vec<Rc<Dao<EvpBits<C, W>>>>,
    //data: Vec<ArrayView1<'a, EvpBits<C, W>>>,
}

pub trait DaoManager<C: BitsContainer, const W: usize> {
    fn new(daos: Vec<Rc<Dao<EvpBits<C, W>>>>) -> Self;
    fn is_mapped(&self, addr: GlobalAddress) -> bool;
    fn table_addr_from_global_addr(&self, addr: &GlobalAddress) -> LocalAddress;
    fn global_addr_from_table_addr(&self, addr: &LocalAddress) -> GlobalAddress;
}

impl<C: BitsContainer, const W: usize> DaoManager<C, W> for DaoStore<C, W> {
    fn new(daos: Vec<Rc<Dao<EvpBits<C, W>>>>) -> Self {
        // let data: Vec<ArrayView1<EvpBits<C, W>>> = daos
        //     .iter()
        //     .map(|dao| dao.clone().get_data())
        //     .collect::<Vec<_>>();

        Self {
            daos: daos,
            //data: data,
        }
    }

    fn is_mapped(&self, addr: GlobalAddress) -> bool {
        let addr = GlobalAddress::as_u32(addr);
        self.daos
            .iter()
            .map(|dao| {
                if addr >= dao.base_addr as u32 && addr < (dao.base_addr + dao.num_data) as u32 {
                    true
                } else {
                    false
                }
            })
            .fold(false, |acc, x| acc || x)
    }

    fn table_addr_from_global_addr(&self, addr: &GlobalAddress) -> LocalAddress {
        todo!()
    }

    fn global_addr_from_table_addr(&self, addr: &LocalAddress) -> GlobalAddress {
        todo!()
    }
}
