use std::collections::LinkedList;
use std::ops::Mul;
use dao::{dao_metadata_from_dir, DaoMetaData};

pub fn sausage<T: Default + Mul>(a: T, b: T) -> T {
    T::default() * a * b
}

trait Stack<T: Default + Mul> {
    fn push(&mut self, v: T);
    fn pop(&mut self) -> T;
}

struct ArrayStack<T> {
    stack: Vec<T>,
}
struct ListStack<T> {
    stack: LinkedList<T>,
}

impl<T> ArrayStack<T> {
    fn new() -> Self {
        Self { stack: Vec::new() }
    }
}

impl<T> ListStack<T> {
    fn new() -> Self {
        Self {
            stack: LinkedList::new(),
        }
    }
}

impl<T: Default + Mul> Stack<T> for ArrayStack<T> {
    fn push(&mut self, v: T) {
        // push operation on array
        self.stack.push(T::default() * v);
    }
    fn pop(&mut self) -> T {
        self.stack.pop().unwrap()
    }
}

impl<T: Default + Mul> Stack<T> for ListStack<T> {
    fn push(&mut self, v: T) {
        self.stack.push_back(T::default() * v);
    }
    fn pop(&mut self) -> T {
        self.stack.pop_back().unwrap()
    }
}

fn do_something<U: Default + Mul, S: Stack<U>>(value: U, mut s: S) {
    s.push(value);
}

fn test(a: i32) {
    let s: Box<dyn Stack<_>> = if a == 0 {
        Box::new(ArrayStack::new())
    } else {
        Box::new(ListStack::new())
    };

    do_something(5, *s);
}

fn main() -> std::io::Result<()> {
    // unused.

    Ok(())
}
