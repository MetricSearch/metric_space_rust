use anyhow::anyhow;
use bits::container::BitsContainer;
use bits::EvpBits;
use dao::Dao;
use utils::address::{GlobalAddress, LocalAddress};

pub struct DaoStore<C: BitsContainer, const W: usize> {
    pub daos: Vec<Dao<EvpBits<C, W>>>,
}

pub trait DaoManager<C: BitsContainer, const W: usize> {
    fn new(daos: Vec<Dao<EvpBits<C, W>>>) -> Self;
    fn is_mapped(&self, addr: &GlobalAddress) -> bool;
    fn table_addr_from_global_addr(&self, addr: &GlobalAddress) -> anyhow::Result<LocalAddress>;
    fn global_addr_from_table_addr(&self, addr: &LocalAddress) -> anyhow::Result<GlobalAddress>;
    fn get_dao(&self, target_addr: &GlobalAddress) -> anyhow::Result<&Dao<EvpBits<C, W>>>;
}

impl<C: BitsContainer, const W: usize> DaoManager<C, W> for DaoStore<C, W> {
    fn new(daos: Vec<Dao<EvpBits<C, W>>>) -> Self {
        log::debug!("Creating DaoManager from ranges: {}", get_ranges(&daos));
        Self { daos }
    }

    fn is_mapped(&self, addr: &GlobalAddress) -> bool {
        let addr = addr.as_u32();
        let result = self
            .daos
            .iter()
            .map(|dao| {
                addr >= dao.base_addr as u32
                    && addr < (dao.base_addr as usize + dao.num_data) as u32
            })
            .fold(false, |acc, x| acc || x);
        result
    }

    /// Returns the offset in the mapped table for a global address of data that is mapped or returns an error.
    fn table_addr_from_global_addr(
        &self,
        target_addr: &GlobalAddress,
    ) -> anyhow::Result<LocalAddress> {
        let mut local_addresses_earlier: u32 = 0;
        let target_addr = target_addr.as_u32();
        for dao in &self.daos {
            let dao_size = dao.num_data as u32;
            if target_addr >= dao.base_addr && target_addr < (dao.base_addr + dao_size) {
                let offset_in_dao = target_addr - dao.base_addr;
                // We have found it.
                let local_addr = offset_in_dao + local_addresses_earlier;
                return Ok(LocalAddress::into(local_addr));
            } else {
                local_addresses_earlier = local_addresses_earlier + dao_size;
            }
        }
        Err(anyhow!(
            "Local Address {} not found in mapping table mapped range: {}",
            target_addr,
            get_ranges(&self.daos)
        ))
    }

    /// Returns the local table address for a global address that is mapped or returns an error.
    fn global_addr_from_table_addr(
        &self,
        target_addr: &LocalAddress,
    ) -> anyhow::Result<GlobalAddress> {
        let target_addr = LocalAddress::as_u32(target_addr);
        let mut addresses_processed = 0;
        for dao in &self.daos {
            if target_addr >= addresses_processed
                && target_addr < addresses_processed + dao.num_data as u32
            {
                // we have found the right dao.
                let offset_in_dao: u32 = target_addr - addresses_processed;
                let result_index = dao.base_addr + offset_in_dao;
                return Ok(GlobalAddress::into(result_index));
            } else {
                addresses_processed = addresses_processed + dao.num_data as u32;
            }
        }
        Err(anyhow!(
            "Global Address {} not found in mapping table",
            target_addr
        ))
    }

    fn get_dao(&self, target_addr: &GlobalAddress) -> anyhow::Result<&Dao<EvpBits<C, W>>> {
        let target_addr = target_addr.as_u32();
        for dao in &self.daos {
            if target_addr >= dao.base_addr && target_addr < dao.base_addr + dao.num_data as u32 {
                // we have found it
                return Ok(dao);
            }
        }
        Err(anyhow!("Didn't find {} in the dao", target_addr))
    }
}

pub fn get_ranges<C: BitsContainer, const W: usize>(daos: &Vec<Dao<EvpBits<C, { W }>>>) -> String {
    daos.iter()
        .map(|dao| {
            let start = dao.base_addr;
            let end = start + dao.embeddings.len() as u32 - 1;
            format!("[{}..{}] (inc)", start, end)
        })
        .collect::<Vec<_>>()
        .join(", ")
}
