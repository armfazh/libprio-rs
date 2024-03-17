// SPDX-License-Identifier: MPL-2.0

//! bt implements a binary tree.
//!

use std::fmt::{Debug, Display};

use bitvec::{order::Lsb0, slice::BitSlice};

/// TT
pub trait TT: Debug + Display + Default {
    // index
    // fn index(&self) -> u8;
}

/// Node
#[derive(Default, Debug)]
struct Node<T: TT> {
    payload: T,
    // level: usize,
    left: Option<Box<Node<T>>>,
    righ: Option<Box<Node<T>>>,
}

impl<T: TT> Node<T> {
    fn new(payload: T) -> Self {
        Self {
            payload,
            left: None,
            righ: None,
        }
    }

    fn new_child(payload: T) -> Option<Box<Self>> {
        Some(Box::new(Node::new(payload)))
    }

    /// insert_bitvec
    fn insert_bitvec(&mut self, payload: T, index: &BitSlice<u8, Lsb0>) {
        if let Some(bit) = index.first() {
            if bit == false {
                match &mut self.left {
                    None => self.left = Self::new_child(payload),
                    Some(n) => n.insert_bitvec(payload, &index[1..]),
                }
            } else {
                match &mut self.righ {
                    None => self.righ = Self::new_child(payload),
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
        if let Some(n) = &self.left {
            out += &n.print_in_order();
        }

        out += &format!("{} ", self.payload);

        if let Some(n) = &self.righ {
            out += &n.print_in_order();
        }

        out
    }
}

impl<T: TT> Display for Node<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let prefix = "  ".repeat(2);
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
        write!(f, "■\n{}", self.root)
    }
}

impl<T: TT> Tree<T> {
    /// new
    pub fn new(root_payload: T) -> Self {
        Self {
            root: Node::new(root_payload),
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
        write!(f, "{:08b}", self.data2)
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

        let mut tree2 = Tree::<MyData>::default();
        for i in 0..3 {
            let j: usize = (3 * i) % 31;
            let j8 = j as u8;
            let index: &BitSlice<u8, _> = j8.view_bits();
            s = tree2.insert_bitvec(my[j].clone(), index);
            println!("j:{} bits:{} {:?}", j, index, s);
        }
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
