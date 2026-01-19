//use bitvec_simd::BitVecSimd;
use dao::csv_dao_loader::dao_from_csv_dir;
use dao::Dao;
use divan::{black_box, Bencher};
use ndarray::Array1;
use std::rc::Rc;

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

#[derive(Clone)]
pub struct Array1I4 {
    data: Array1<u8>,
    len: usize, // logical number of i4 elements
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
        if idx % 2 == 0 {
            a
        } else {
            b
        }
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

pub struct Array1I4Iter<'a> {
    data: &'a [u8],
    len: usize,
    idx: usize,
}

impl<'a> Iterator for Array1I4Iter<'a> {
    type Item = i8;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.len {
            return None;
        }

        let byte = self.data[self.idx / 2];
        let nibble = if self.idx % 2 == 0 {
            (byte & 0x0F) as i8
        } else {
            ((byte >> 4) & 0x0F) as i8
        };

        // sign extend i4 → i8
        let value = if nibble & 0x08 != 0 {
            nibble | !0x0F
        } else {
            nibble
        };

        self.idx += 1;
        Some(value)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len - self.idx;
        (remaining, Some(remaining))
    }
}

impl<'a> IntoIterator for &'a Array1I4 {
    type Item = i8;
    type IntoIter = Array1I4Iter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        Array1I4Iter {
            data: self.data.as_slice().expect("contiguous storage"),
            len: self.len,
            idx: 0,
        }
    }
}

pub fn euc_4bit(a: &Array1I4, b: &Array1I4) -> f32 {
    a.into_iter()
        .zip(b.into_iter())
        .map(|(a, b)| (a - b).pow(2) as f32)
        .sum()
}

pub fn get_max(data: &Array1<f32>) -> f32 {
    return data.iter().cloned().reduce(f32::max).unwrap();
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

fn main() {
    divan::main();
}

#[divan::bench]
fn bench(bencher: Bencher) {
    // bencher: Bencher

    let num_queries = 10_000;
    let num_data = 1_000_000 - num_queries;

    let dao: Rc<Dao<Array1<f32>>> = Rc::new(
        dao_from_csv_dir(
            "/Volumes/Data/RUST_META/mf_dino2_csv/",
            num_data,
            num_queries,
        )
        .unwrap(),
    );

    let data_f32 = dao.get_datum(0);
    let max_f32 = get_max(data_f32);
    let query_f32 = dao.get_query(0);

    let query: Array1I4 = Array1I4::from_i8(&to_i8_array(query_f32, max_f32));
    let data: Array1I4 = Array1I4::from_i8(&to_i8_array(data_f32, max_f32));

    bencher.bench(|| {
        let res = euc_4bit(black_box(&query), black_box(&data));
        black_box(res);
    });
}
