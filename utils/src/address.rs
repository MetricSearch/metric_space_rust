#[derive(PartialEq)]
pub struct LocalAddress(u32);

#[derive(PartialEq, Debug)]
pub struct GlobalAddress(u32);

impl GlobalAddress {
    pub fn into(x: u32) -> Self {
        GlobalAddress(x)
    }
    pub fn as_u32(x: GlobalAddress) -> u32 {
        x.0
    }

    pub fn as_usize(x: GlobalAddress) -> usize {
        x.0 as usize
    }

    pub fn combine(&self, sim: f32) -> u64 {
        // ((sim as u32) as u64)
        (f32::to_bits(sim) as u64) | ((self.0 as u64) << 32)
    }
}

impl LocalAddress {
    pub fn into(x: u32) -> Self {
        LocalAddress(x)
    }

    pub fn as_u32(&self) -> u32 {
        self.0
    }

    pub fn as_usize(&self) -> usize {
        self.0 as usize
    }
}
