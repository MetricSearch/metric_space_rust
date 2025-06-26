use anyhow::anyhow;
use bits::container::BitsContainer;
use bits::EvpBits;
use dao::Dao;
use utils::address::{GlobalAddress, LocalAddress};

pub struct DaoStore<C: BitsContainer, const W: usize> {
    daos: Vec<Dao<EvpBits<C, W>>>,
}

pub trait DaoManager<C: BitsContainer, const W: usize> {
    fn new(daos: Vec<Dao<EvpBits<C, W>>>) -> Self;
    fn is_mapped(&self, addr: GlobalAddress) -> bool;
    fn table_addr_from_global_addr(&self, addr: &GlobalAddress) -> LocalAddress;
    fn global_addr_from_table_addr(&self, addr: &LocalAddress) -> anyhow::Result<GlobalAddress>;
}

impl<C: BitsContainer, const W: usize> DaoManager<C, W> for DaoStore<C, W> {
    fn new(daos: Vec<Dao<EvpBits<C, W>>>) -> Self {
        Self { daos }
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

    fn global_addr_from_table_addr(
        &self,
        target_addr: &LocalAddress,
    ) -> anyhow::Result<GlobalAddress> {
        let target_addr = LocalAddress::as_u32(target_addr) as usize;
        let mut addresses_processed = 0;
        for dao in &self.daos {
            if target_addr > addresses_processed && target_addr < addresses_processed + dao.num_data
            {
                // we have found the right dao.
                let offset_in_dao: usize = target_addr - addresses_processed;
                let result_index = dao.base_addr + offset_in_dao;
                let result_index = result_index
                    .try_into()
                    .unwrap_or_else(|_| panic!("cannot convert into u32 from usize"));
                return Ok(GlobalAddress::into(result_index));
            } else {
                addresses_processed = addresses_processed + dao.num_data;
            }
        }
        Err(anyhow!("No such global address {}", target_addr))
    }
}
