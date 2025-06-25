#[derive(PartialEq)]
pub struct LocalAddress(u32);

#[derive(PartialEq)]
pub struct GlobalAddress(u32);

impl GlobalAddress {
    pub fn into(x: u32) -> Self {
        GlobalAddress(x)
    }
    pub fn as_u32(x: GlobalAddress) -> u32 {
        x.0
    }

    pub fn combine(&self, sim: f32) -> u64 {
        // ((sim as u32) as u64)
        (f32::to_bits(sim) as u64) | ((self.0 as u64) << 32)
    }
}

pub trait TableAddress {
    fn new(table_index: u32) -> Self;
    fn as_usize(&self) -> usize;
}

impl TableAddress for LocalAddress {
    fn new(index: u32) -> Self {
        LocalAddress(index)
    }

    fn as_usize(&self) -> usize {
        self.0 as usize
    }
}
