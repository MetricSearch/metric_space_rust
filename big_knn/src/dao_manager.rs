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
    fn is_mapped(&self, addr: GlobalAddress, row: usize) -> bool;
    fn table_addr_from_global_addr(&self, addr: &GlobalAddress) -> anyhow::Result<LocalAddress>;
    fn global_addr_from_table_addr(&self, addr: &LocalAddress) -> anyhow::Result<GlobalAddress>;
    fn get_dao(&self, target_addr: &GlobalAddress) -> anyhow::Result<&Dao<EvpBits<C, W>>>;
}

impl<C: BitsContainer, const W: usize> DaoManager<C, W> for DaoStore<C, W> {
    fn new(daos: Vec<Dao<EvpBits<C, W>>>) -> Self {
        Self { daos }
    }

    fn is_mapped(&self, addr: GlobalAddress, row: usize) -> bool {
        let addr = GlobalAddress::as_u32(addr);
        let result = self
            .daos
            .iter()
            .map(|dao| {
                addr >= dao.base_addr as u32
                    && addr < (dao.base_addr as usize + dao.num_data) as u32
            })
            .fold(false, |acc, x| acc || x);

        if result == false {
            println!("Is mapped false for addr {} at row {}", addr, row);
        }
        result
    }

    /// Returns the global address for a local table address for data that is mapped or returns an error.
    fn table_addr_from_global_addr(
        &self,
        target_addr: &GlobalAddress,
    ) -> anyhow::Result<LocalAddress> {
        let mut table_index: u32 = 0;
        let target_addr = GlobalAddress::as_u32(*target_addr);
        for dao in &self.daos {
            if target_addr >= dao.base_addr && target_addr < dao.base_addr + dao.num_data as u32 {
                // We have found it.
                let difference_from_dao_base: u32 = (target_addr as u32 - dao.base_addr)
                    .try_into()
                    .unwrap_or_else(|_| panic!("cannot convert into u32 from usize"));
                return Ok(LocalAddress::into(table_index + difference_from_dao_base));
            } else {
                table_index = table_index + dao.num_data as u32;
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
                let result_index = result_index
                    .try_into()
                    .unwrap_or_else(|_| panic!("cannot convert into u32 from usize"));
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
        let target_addr = GlobalAddress::as_u32(*target_addr);
        for dao in &self.daos {
            if target_addr >= dao.base_addr && target_addr < dao.base_addr + dao.num_data as u32 {
                // we have found it
                return Ok(dao);
            }
        }
        Err(anyhow!("Didn't find the dao"))
    }
}

fn get_ranges<C: BitsContainer, const W: usize>(daos: &Vec<Dao<EvpBits<C, { W }>>>) -> String {
    daos.iter()
        .map(|dao| {
            let start = dao.base_addr;
            let end = start + dao.embeddings.len() as u32;
            format!("<{}..{}>", start, end)
        })
        .collect::<Vec<_>>()
        .join(", ")
}
