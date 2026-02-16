use anyhow::Result;
use ndarray::Array1;
use rand::random;
use rayon::prelude::*;
use std::time::Instant;

#[inline]
fn clamp_i4(x: i8) -> i8 {
    x.clamp(-8, 7)
}

#[inline]
fn pack_i4_pair(a: i8, b: i8) -> u8 {
    let a = (clamp_i4(a) & 0x0F) as u8;
    let b = (clamp_i4(b) & 0x0F) as u8;
    (b << 4) | a
}

#[inline]
fn unpack_i4(x: u8) -> (i8, i8) {
    let lo = (x & 0x0F) as i8;
    let hi = ((x >> 4) & 0x0F) as i8;

    // sign extend
    let lo = if lo & 0x08 != 0 { lo | !0x0F } else { lo };
    let hi = if hi & 0x08 != 0 { hi | !0x0F } else { hi };

    (lo, hi)
}

pub struct Array1I4View<'a> {
    parent: &'a Array1I4,
    start: usize,
    len: usize,
}

#[derive(Clone)]
pub struct Array1I4 {
    data: Array1<u8>,
    len: usize, // logical number of i4 elements
}

impl<'a> Array1I4View<'a> {
    pub fn iter(&'a self) -> Array1I4Iter<'a> {
        Array1I4Iter { view: self, pos: 0 }
    }
}

impl<'a> Array1I4View<'a> {
    pub fn get(&self, idx: usize) -> i8 {
        assert!(idx < self.len);

        let byte = self.parent.data[idx / 2];

        let val = if idx % 2 == 0 {
            byte & 0x0F
        } else {
            (byte >> 4) & 0x0F
        };

        // sign extend 4-bit value to i8
        ((val as i8) << 4) >> 4
    }
}

pub struct Array1I4Iter<'a> {
    view: &'a Array1I4View<'a>,
    pos: usize,
}

impl<'a> Iterator for Array1I4Iter<'a> {
    type Item = i8;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.view.len {
            return None;
        }

        let value = self.view.get(self.pos);
        self.pos += 1;
        Some(value)
    }
}

impl Array1I4 {
    pub fn slice(&self, start: usize, end: usize) -> Array1I4View<'_> {
        assert!(start <= end);
        assert!(end <= self.len);

        Array1I4View {
            parent: self,
            start,
            len: end - start,
        }
    }
}

impl Array1I4 {
    pub fn from_i8(src: &Array1<i8>) -> Self {
        let len = src.len();
        let packed_len = (len + 1) / 2;

        let mut data = Array1::<u8>::zeros(packed_len);

        for i in 0..packed_len {
            let a = src[2 * i];
            let b = if 2 * i + 1 < len { src[2 * i + 1] } else { 0 };
            data[i] = pack_i4_pair(a, b);
        }

        Self { data, len }
    }
}

impl Array1I4 {
    #[inline]
    pub fn get(&self, idx: usize) -> i8 {
        assert!(idx < self.len);
        let byte = self.data[idx / 2];
        let (a, b) = unpack_i4(byte);
        if idx % 2 == 0 { a } else { b }
    }
}

impl Array1I4 {
    pub fn to_i8(&self) -> Array1<i8> {
        let mut out = Array1::<i8>::zeros(self.len);
        for i in 0..self.len {
            out[i] = self.get(i);
        }
        out
    }
}

pub fn euc_4bit(a: &Array1I4View, b: Array1I4View) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(a, b)| (a - b).pow(2) as f32)
        .sum::<f32>()
}

pub fn to_i8_array(array: &Array1<f32>, max_f32: f32) -> Array1<i8> {
    array.mapv(|x| {
        let value = x / max_f32;

        if value.is_nan() {
            // this will never happen
            0
        } else {
            (value * i8::MAX as f32)
                .round()
                .clamp(i8::MIN as f32, i8::MAX as f32) as i8
        }
    })
}

fn main() -> Result<()> {
    let num_queries = 100;
    let num_data = 1_000_000;

    //----------------

    for dims in [100, 384, 500, 768] {
        do_experiment(num_queries, num_data, dims)
    }

    Ok(())
}

fn do_experiment(num_queries: usize, num_data: usize, dims: usize) {
    let queries_f32 = Array1::from_iter((0..dims * num_queries).map(|_| random::<f32>()));
    let data_f32 = Array1::from_iter((0..dims * num_data).map(|_| random::<f32>()));

    let max_f32 = data_f32.iter().copied().map(|x| x.abs()).fold(f32::NEG_INFINITY, f32::max);

    let queries: Array1I4 = Array1I4::from_i8(&to_i8_array(&queries_f32, max_f32));
    let data = Array1I4::from_i8(&to_i8_array(&data_f32, max_f32));

    let now = Instant::now();

    // Do a brute force of queries against the data
    let four_bit_distances = generate_4bit_dists(queries, data, num_queries, num_data, dims);

    let after = Instant::now();
    println!("Last distance is {:?}", four_bit_distances.iter().flatten().last());


    println!(
        "Time per 4bit {} dim query 1_000_000 dists: {} ns",
        dims,
        ((after - now).as_nanos() as f64) / num_queries as f64
    );
}

fn generate_4bit_dists(
    queries: Array1I4,
    data: Array1I4,
    num_queries: usize,
    num_data: usize,
    dims: usize,
) -> Vec<Vec<f32>> {
    (0..num_queries)
        .par_bridge()
        .map(|q_index| {
            (0..num_data)
                .map(|data_index| {
                    let q = queries.slice(q_index * dims, (q_index * dims) + dims);
                    let d = data.slice(data_index * dims, (data_index * dims) + dims);
                    euc_4bit(&q, d)
                })
                .collect::<Vec<f32>>()
        })
        .collect::<Vec<Vec<f32>>>()
}
