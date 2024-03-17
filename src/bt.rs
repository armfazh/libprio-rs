// SPDX-License-Identifier: MPL-2.0

//! bt implements a binary tree.
//!

use std::fmt::{Debug, Display};

use bitvec::{
    order::Lsb0,
    slice::{BitSlice, Iter},
};

/// TT
pub trait TT: Debug + Display + Default {
    // index
    // fn index(&self) -> u8;
}

/// Node
#[derive(Default, Debug)]
struct Node<T: TT> {
    payload: T,
    level: usize,
    left: Option<Box<Node<T>>>,
    righ: Option<Box<Node<T>>>,
}

#[derive(Debug)]
struct IterLeft<'a, T: TT> {
    next: Option<&'a Node<T>>,
}

impl<'a, T: TT> IterLeft<'a, T> {
    fn sig(&mut self) -> Option<&'a T> {
        self.next.map(|node| {
            self.next = node.left.as_deref();
            &node.payload
        })
    }
}

#[derive(Debug)]
struct IterMutLeft<'a, T: TT> {
    next: Option<&'a mut Node<T>>,
}

impl<'a, T: TT> IterMutLeft<'a, T> {
    fn sig(&mut self) -> Option<&'a mut T> {
        self.next.take().map(|node| {
            self.next = node.left.as_deref_mut();
            &mut node.payload
        })
    }
}

impl<T: TT> Node<T> {
    fn iter(&self) -> IterLeft<'_, T> {
        IterLeft { next: Some(self) }
    }

    fn iter_mut(&mut self) -> IterMutLeft<'_, T> {
        IterMutLeft { next: Some(self) }
    }

    fn new(payload: T, level: usize) -> Self {
        Self {
            payload,
            level,
            left: None,
            righ: None,
        }
    }

    fn new_child(payload: T, level: usize) -> Option<Box<Self>> {
        Some(Box::new(Node::new(payload, level)))
    }

    /// insert_bitvec
    fn insert_bitvec(&mut self, payload: T, index: &BitSlice<u8, Lsb0>) {
        if let Some(bit) = index.first() {
            if bit == false {
                match &mut self.left {
                    None => self.left = Self::new_child(payload, self.level + 1),
                    Some(n) => n.insert_bitvec(payload, &index[1..]),
                }
            } else {
                match &mut self.righ {
                    None => self.righ = Self::new_child(payload, self.level + 1),
                    Some(n) => n.insert_bitvec(payload, &index[1..]),
                };
            }
        }
    }

    /// insert
    // fn insert(&mut self, py: T) {
    //     let node_index = self.payload.index();
    //     let py_index = py.index();

    //     if py_index < node_index {
    //         match &mut self.left {
    //             None => self.left = Node::new(py, self.level + 1),
    //             Some(n) => return n.insert(py),
    //         }
    //     } else if py_index > node_index {
    //         match &mut self.righ {
    //             None => self.righ = Node::new(py, self.level + 1),
    //             Some(n) => return n.insert(py),
    //         };
    //     }
    // }

    /// print_in_order (recursive)
    fn print_in_order(&self) -> String {
        let mut out = String::new();

        out += &match &self.left {
            None => String::from("X "),
            Some(n) => n.print_in_order(),
        };

        out += &format!("{} ", self.payload);

        out += &match &self.righ {
            None => String::from("X "),
            Some(n) => n.print_in_order(),
        };

        out
    }

    /// print_pre_order (recursive)
    fn print_pre_order(&self) -> String {
        let mut out = String::new();
        out += &format!("{} ", self.payload);

        out += &match &self.left {
            None => String::from("L "),
            Some(n) => n.print_pre_order(),
        };

        out += &match &self.righ {
            None => String::from("R "),
            Some(n) => n.print_pre_order(),
        };

        out
    }
}

impl<T: TT> Display for Node<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let prefix = "  ".repeat(self.level);
        write!(f, "{}└node: {}\n", prefix, self.payload)?;
        match &self.left {
            None => write!(f, "{}└left: <None>\n", prefix)?,
            Some(n) => write!(f, "{}└left:\n{}", prefix, &n)?,
        };
        match &self.righ {
            None => write!(f, "{}└righ: <None>\n", prefix)?,
            Some(n) => write!(f, "{}└righ:\n{}", prefix, &n)?,
        };

        Ok(())
    }
}

/// Tree
#[derive(Default, Debug)]
pub struct Tree<T: TT> {
    root: Node<T>,
}

impl<T: TT> Display for Tree<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "\n■\n{}", self.root)
    }
}

impl<T: TT> Tree<T> {
    /// new
    pub fn new(root_payload: T) -> Self {
        Self {
            root: Node::new(root_payload, 0),
        }
    }

    /// insert_bitvec
    pub fn insert_bitvec(&mut self, payload: T, index: &BitSlice<u8, Lsb0>) {
        self.root.insert_bitvec(payload, index)
    }

    /// insert
    // pub fn insert(&mut self, py: T) {
    //     match &mut self.0 {
    //         None => self.0 = Node::new(py, 0),
    //         Some(node) => node.insert(py),
    //     }
    // }

    /// in_order
    pub fn in_order(&self) -> String {
        self.root.print_in_order()
    }
    /// in_order
    pub fn pre_order(&self) -> String {
        self.root.print_pre_order()
    }
}

/// MyData
#[derive(Default, Debug, Clone)]
pub struct MyData {
    // data1: String,
    data2: u8,
    // data3: Vec<u8>,
    // data4: [u8; 10],
}
impl MyData {
    /// new
    pub fn new(data2: u8) -> Self {
        Self { data2 }
    }
}

impl TT for MyData {
    // fn index(&self) -> u8 {
    //     self.data2
    // }
}

impl Display for MyData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.data2)
    }
}

#[cfg(test)]
mod test {
    use bitvec::{bitvec, prelude::Lsb0, slice::BitSlice, view::BitView};

    use crate::bt::{MyData, Tree, VERSION};

    #[test]
    fn devel() {
        let root = MyData::new(99);
        let tree = Tree::new(root);
        let mut my = Vec::new();
        for i in 0..32 {
            my.push(MyData::new(i as u8));
        }

        println!("{}", tree);

        let index = bitvec![0, 1, 1, 0, 1, 0, 1, 1];
        println!(">index {}", index);

        let mut s;
        // s = root.insert(my[10].clone());
        // println!("{:?} {}", s, root);
        // s = root.insert(my[7].clone());
        // println!("{:?} {}", s, root);
        // s = root.insert(my[13].clone());
        // println!("{:?} {}", s, root);
        // s = root.insert(my[14].clone());
        // println!("{:?} {}", s, root);
        // s = root.insert(my[14].clone());
        // println!("{:?} {}", s, root);

        // for i in 0..31 {
        //     let j = (3 * i + 18) % 31;
        //     s = root.insert(my[j].clone());
        //     println!("i:{} {:?}", j, s);
        // }
        // println!("> {}", root.in_order());

        let root = MyData::new(99);
        let mut tree2 = Tree::<MyData>::new(root);
        for i in 0..3 {
            let j: usize = (3 * i + 4) % 31;
            let j8 = j as u8;
            let index: &BitSlice<u8, _> = j8.view_bits();
            s = tree2.insert_bitvec(my[j].clone(), index);
            println!("j:{} bits:{} {:?}", j, index, s);
        }
        println!("> {}", tree2);
        println!("> {}", tree2.pre_order());

        let mut it = tree2.root.iter();
        println!("> {:?}", it);
        println!("> {:?}", it.sig());
        println!("> {:?}", it.sig());
        println!("> {:?}", it.sig());
        println!("> {:?}", it.sig());

        let mut itm = tree2.root.iter_mut();
        println!("> {:?}", itm);
        println!("> {:?}", itm.sig());
        *itm.sig().unwrap() = MyData::new(77);
        println!("> {:?}", itm.sig());
        println!("> {:?}", itm.sig());

        println!("> {}", tree2);
        assert!(true);
    }

    //
    //
    //
    //
    //
    //
    //

    #[test]
    fn version() {
        assert_eq!(VERSION, 10);
    }
}

// fn insert() {
//     let input: &[u8] = &[0x36];
//     let bits = BitVec::<u8, Lsb0>::from_slice(input);

//     let root = Root::<MyData>::default();

//     for i in bits.into_iter() {
//         if i == true {}
//     }
// }

//
//
//
//
//
//
//
//
//
//
//
//
//
//
//

/// Version is.
const VERSION: usize = 10;

/// v is
pub fn v() -> usize {
    VERSION
}
