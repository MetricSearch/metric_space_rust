// RPTree impl
// al

use dao::Dao;
use itertools::Itertools;
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};
use std::cmp::max;
use std::rc::Rc;
use ndarray::{Array, Array1, ArrayBase, Ix1, ViewRepr};

const SEED: [u8; 32] = [
    1, 34, 53, 43, 111, 12, 65, 67, 9, 32, 53, 41, 49, 12, 66, 67, 6, 33, 51, 91, 44, 13, 50, 69,
    8, 39, 55, 23, 37, 112, 72, 5,
];

/***************************************************************************************************
                                             RPNode
***************************************************************************************************/

pub struct RpNode {
    pub pivot: Array1<f32>,   // The pivot for this node
    pub payload: Vec<usize>, // A vec of indices into the vectors structure
    pub split_value: f32,    // the split value that is used to subdivide the data into children
    pub left: Option<Box<Self>>,
    pub right: Option<Box<Self>>,
}

impl RpNode {
    pub fn new(dao: Rc<Dao>, rng: &mut ChaCha8Rng) -> Self {
        let payload: Vec<usize> = vec![];
        //let distribution = Normal::new(0.0, 1.0).unwrap();
        Self {
            pivot: make_pivot2(dao, rng), //make_pivot(dim,distribution),
            payload: payload,
            split_value: 0.0,
            left: None,
            right: None,
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.left.is_none() && self.right.is_none()
    }

    pub fn add(
        &mut self,
        index: usize,
        max_load: usize,
        dim: usize,
        dao: Rc<Dao>,
        rng: &mut ChaCha8Rng,
    ) {
        // call the private insert method with the tree root
        self.do_insert(index, max_load, dim, dao, rng);
    }

    fn do_insert(
        &mut self,
        index: usize,
        max_load: usize,
        dim: usize,
        dao: Rc<Dao>,
        rng: &mut ChaCha8Rng,
    ) {
        // should not be pub?
        if self.is_leaf() {
            if self.payload.len() >= max_load {
                // perform the split
                self.left = Some(Box::new(RpNode::new(dao.clone(), rng)));
                self.right = Some(Box::new(RpNode::new(dao.clone(), rng)));
                self.redistribute(dao.clone());
                // no return - still need to walk the tree more since current is no longer a leaf
            } else {
                self.payload.push(index);
            }
        }
        // Walk the tree to get to the leaves
        let data_to_add = dao.get_datum(index);
        let dist = dot_product(&self.pivot.view(), data_to_add);
        if dist <= self.split_value {
            if let Some(left) = self.left.as_mut() {
                left.do_insert(index, max_load, dim, dao.clone(), rng);
            }
        } else {
            if let Some(right) = self.right.as_mut() {
                right.do_insert(index, max_load, dim, dao.clone(), rng);
            }
        }
    }

    pub fn depth(&self) -> usize {
        let right_depth = match self.right {
            Some(ref right) => right.depth(),
            None => 0,
        };
        let left_depth = match self.left {
            Some(ref left) => left.depth(),
            None => 0,
        };
        1 + max(right_depth, left_depth)
    }

    pub fn do_lookup(&self, query: ArrayBase<ViewRepr<&f32>,Ix1>, dao: Rc<Dao>) -> Option<Vec<usize>> {
        if self.is_leaf() {
            Some(self.payload.clone())
        } else {
            let dp = dot_product(&self.pivot.view(), query.view());
            if dp <= self.split_value {
                if let Some(left) = &self.left {
                    left.do_lookup(query, dao)
                } else {
                    None
                }
            } else {
                if let Some(right) = &self.right {
                    right.do_lookup(query, dao)
                } else {
                    None
                }
            }
        }
    }

    /**************  private functions **************/

    fn pretty_print(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self.payload)?;
        if let Some(left) = &self.left {
            write!(f, "\n\tL: {}", left)?;
        }
        if let Some(right) = &self.right {
            write!(f, "\n\tR: {}", right)?;
        }
        return Result::Ok(());
    }

    fn redistribute(&mut self, dao: Rc<Dao>) {
        let prods = self
            .payload
            .iter()
            .map(|id| {
                dot_product(
                    &self.pivot.view(),
                    dao.get_datum(*id),
                )
            })
            .collect::<Vec<f32>>(); // calculate the dot products to each data from the pivot
                                    // next calculate the split position

        tracing::info!("{:?}", self.payload);
        tracing::info!("{:?}", prods);
        let split_value = self.unif_split(&prods);
        tracing::info!("Split {}", split_value);

        // redistribute the data to the children from the root payload
        self.split_value = split_value; // keep for later

        self.payload.iter().enumerate().for_each(|(index, id)| {
            if prods[index] <= split_value {
                if let Some(left) = self.left.as_mut() {
                    tracing::info!("id {} prod {} going left", id, prods[index]);
                    left.payload.push(*id);
                }
            } else {
                if let Some(right) = self.right.as_mut() {
                    tracing::info!("id {} prod {} going right", id, prods[index]);
                    right.payload.push(*id);
                }
            }
        });
        self.set_not_root();
    }

    fn unif_split(&mut self, products: &Vec<f32>) -> f32 {
        // This took me a long time to write, so I am not deleting it!
        //let min = products.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        //let max = products.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        //StdRng::from_seed(SEED).gen_range(min..max) as f32

        let mut sorted = products.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        tracing::info!("Sorted: {:?}", sorted);
        sorted[(sorted.len() - 1) / 2] // return the middle valye of the products
    }
    fn set_not_root(&mut self) {
        self.payload.clear();
    }
}

impl std::fmt::Display for RpNode {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.pretty_print(f)
    }
}

impl std::fmt::Debug for RpNode {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.pretty_print(f)
        // OR
        //write!( f, "{self}" ) // calls Display
    }
}

pub fn make_pivot2(dao: Rc<Dao>, rng: &mut ChaCha8Rng) -> Array1<f32> {
    let index = rng.gen_range(0..dao.num_data);
    tracing::info!("** PIVOT ** : {}", index);
    dao.get_datum(index).into_owned()
}

fn make_pivot(dim: usize, distribution: Normal<f32>) -> Vec<f32> {
    // IF WE USE THIS NEED TO ADD rng
    let values: Vec<f32> = (0..dim) // for all the required dimensions
        .map(|_| distribution.sample(&mut StdRng::from_seed(SEED))) // map randoms in range 0..1
        .collect(); // put them all in a vector
    let mag_squared: f32 = values
        .iter() // take all the elements in the Vector
        .map(|x| (x * x)) // square each of them
        .sum(); // and sum them up

    let mag = mag_squared.sqrt();

    let normed = values
        .iter() // take all the values
        .map(|x| x / mag) // divide by the mag
        .collect(); // collect them up into a vec

    normed // return the norm
}

/***************************************************************************************************
                                             RPTree
***************************************************************************************************/

//#[derive(Debug)]
pub struct RPTree {
    dao: Rc<Dao>,
    root: Option<Box<RpNode>>,
    max_load: usize,
    dim: usize,
    rng: ChaCha8Rng,
}

impl RPTree {
    pub(crate) fn depth(&self) -> usize {
        match self.root {
            Some(ref root) => root.depth(),
            None => 0,
        }
    }
}

impl RPTree {
    pub fn new(max_load: usize, dao: Rc<Dao>, use_as_seed: u64) -> Self {
        let dim = dao.get_dim();
        let rng = rand_chacha::ChaCha8Rng::seed_from_u64(use_as_seed * 142);
        Self {
            root: Option::None,
            dao,
            max_load,
            dim,
            rng,
        }
    }

    pub fn lookup(&self, query: ArrayBase<ViewRepr<&f32>,Ix1>) -> Option<Vec<usize>> {
        self.root
            .as_ref()
            .and_then(|node| node.do_lookup(query, self.dao.clone()))
    }

    // pub fn lookup(&self, query: &[f32]) -> Result<Option<Vec<usize>>> {
    //     self.root.as_ref()
    //         .ok_or_else(|| anyhow!("Tree is empty"))
    //         .map(|node| node.do_lookup(query, self.dao.clone()))
    //     // this is the same as this:
    //     // match &self.root {
    //     //     None => Result::Err(anyhow!("Tree is empty")),
    //     //     Some(node) => { Ok( node.do_lookup(query,self.dao.clone()) ) }
    //     // }
    // }

    pub fn add(&mut self, table_index: usize) {
        match &mut self.root {
            None => {
                let mut node = RpNode::new(self.dao.clone(), &mut self.rng);
                node.payload.push(table_index);
                self.root = Some(Box::new(node));
            }
            Some(node) => {
                node.add(
                    table_index,
                    self.max_load,
                    self.dim,
                    self.dao.clone(),
                    &mut self.rng,
                );
            }
        }
    }

    /**************  private functions **************/

    fn populate(&mut self) {
        tracing::info!("Data len {}", self.dao.data_len());
        for i in 0..self.dao.data_len() {
            if i % 100_000 == 0 {
                tracing::info!("Adding data {i}");
            }
            self.add(i);
        }
    }
}

impl std::fmt::Display for RPTree {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if let Some(node) = &self.root {
            write!(f, "\t{:?}", node)
        } else {
            write!(f, "\tempty")
        }
    }
}

impl std::fmt::Debug for RPTree {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if let Some(node) = &self.root {
            write!(f, "\t{:?}", node)
        } else {
            write!(f, "\tempty")
        }
    }
}

/***************************************************************************************************
                                             RPForest
***************************************************************************************************/

#[derive(Debug)]
pub struct RPForest {
    trees: Vec<RPTree>,
}

impl RPForest {
    pub fn new(num_trees: usize, max_load: usize, dao: Rc<Dao>) -> Self {
        let mut trees = vec![];
        for use_as_seed in 0..num_trees {
            let tree = RPTree::new(max_load, dao.clone(), use_as_seed as u64);
            trees.push(tree);
        }

        let mut this = Self { trees };
        this.populate(dao.clone());
        this
    }

    pub fn populate(&mut self, dao: Rc<Dao>) {
        for i in 0..dao.data_len() {
            if i % 100_000 == 0 {
                for j in 0..self.trees.len() {
                    self.trees[j].add(i);
                }
                tracing::info!("Adding data {i}");
            }
            self.add(i);
        }
    }

    pub fn add(&mut self, index: usize) {
        self.trees.iter_mut().for_each(|tree| tree.add(index));
    }

    pub fn lookup(&self, query: ArrayBase<ViewRepr<&f32>,Ix1>) -> Vec<usize> {
        self.trees
            .iter()
            .filter_map(|tree| tree.lookup(query))
            .flatten()
            .unique()
            .collect()
    }
}

/***************************************************************************************************
                                             Utility functions
***************************************************************************************************/

fn dot_product(p0: &ArrayBase<ViewRepr<&f32>, Ix1>, p1: ArrayBase<ViewRepr<&f32>, Ix1>) -> f32 {
    assert!(p0.len() == p1.len());
    p0.iter().zip(p1.iter()).map(|(x, y)| x * y).sum()

    // equivalent to:
    // let mut result = 0.0;
    // for i in 0..p0.len() {
    //     result += p0[i] * p1[i];
    // }
    // result
}
