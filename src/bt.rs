// SPDX-License-Identifier: MPL-2.0

//! bt implements a binary tree.
//!

use std::fmt::{Debug, Display};

use bitvec::{order::Msb0, slice::BitSlice};

/// TT
pub trait TT: Debug + Display {}

/// Node
#[derive(Debug)]
struct Node<T: TT> {
    payload: T,
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

    /// insert (iterative)
    pub fn insert_iter(&mut self, index: &BitSlice<u8, Msb0>, payload: T) {
        let mut itm = Some(self);

        for bit in index.iter() {
            let node = itm.take().unwrap();

            if bit == false {
                match node.left {
                    None => {
                        node.left = Some(Box::new(Node::new(payload)));
                        break;
                    }
                    Some(_) => itm = node.left.as_deref_mut(),
                };
            } else {
                match node.righ {
                    None => {
                        node.righ = Some(Box::new(Node::new(payload)));
                        break;
                    }
                    Some(_) => itm = node.righ.as_deref_mut(),
                };
            }
        }
    }

    /// insert_bitvec (recursive)
    fn insert_rec(&mut self, index: &BitSlice<u8, Msb0>, payload: T) {
        if let Some(bit) = index.first() {
            if bit == false {
                match &mut self.left {
                    None => self.left = Some(Box::new(Node::new(payload))),
                    Some(node) => node.insert_rec(&index[1..], payload),
                }
            } else {
                match &mut self.righ {
                    None => self.righ = Some(Box::new(Node::new(payload))),
                    Some(node) => node.insert_rec(&index[1..], payload),
                };
            }
        }
    }

    /// print_in_order (recursive)
    fn print_in_order_rec(&self) -> String {
        let mut out = String::new();

        out += &match &self.left {
            None => String::from("L "),
            Some(n) => n.print_in_order_rec(),
        };

        out += &format!("{} ", self.payload);

        out += &match &self.righ {
            None => String::from("R "),
            Some(n) => n.print_in_order_rec(),
        };

        out
    }
    /// print_pre_order (recursive)
    fn print_pre_order_rec(&self) -> String {
        let mut out = String::new();
        out += &format!("{} ", self.payload);

        out += &match &self.left {
            None => String::from("L "),
            Some(n) => n.print_pre_order_rec(),
        };

        out += &match &self.righ {
            None => String::from("R "),
            Some(n) => n.print_pre_order_rec(),
        };

        out
    }

    /// print_pre_order (iterative)
    fn print_pre_order_iter(&self) -> String {
        struct Item<'a, T: TT> {
            is_righ: bool,
            value: Option<&'a Node<T>>,
        }

        let mut stack = Vec::<Item<T>>::new();

        stack.push(Item {
            is_righ: false,
            value: Some(self),
        });

        let mut out = String::new();
        while let Some(elem) = stack.pop() {
            if let Some(node) = elem.value {
                out += &format!("{} ", node.payload);

                stack.push(Item {
                    is_righ: true,
                    value: node.righ.as_deref(),
                });

                stack.push(Item {
                    is_righ: false,
                    value: node.left.as_deref(),
                });
            } else {
                out += if elem.is_righ { "R " } else { "L " };
            }
        }

        out
    }

    fn print_tree_iter(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        struct Item<'a, T: TT> {
            name: String,
            level: usize,
            value: Option<&'a Node<T>>,
        }

        let mut stack = Vec::<Item<T>>::new();

        stack.push(Item {
            name: String::default(),
            level: 0,
            value: Some(self),
        });

        while let Some(mut elem) = stack.pop() {
            if let Some(node) = elem.value {
                let prefix = "  ".repeat(elem.level);

                if !elem.name.is_empty() {
                    elem.name.push('\n')
                }

                writeln!(f, "{}{}└node: {}", elem.name, prefix, node.payload)?;

                stack.push(Item {
                    name: format!("{}└righ:", prefix),
                    level: elem.level + 1,
                    value: node.righ.as_deref(),
                });

                stack.push(Item {
                    name: format!("{}└left:", prefix),
                    level: elem.level + 1,
                    value: node.left.as_deref(),
                });
            } else {
                writeln!(f, "{} <None>", elem.name)?;
            }
        }

        Ok(())
    }

    fn print_tree_rec(&self, f: &mut std::fmt::Formatter<'_>, level: usize) -> std::fmt::Result {
        let prefix = "  ".repeat(level);

        writeln!(f, "{}└node: {}", prefix, self.payload)?;

        match &self.left {
            Some(node) => {
                writeln!(f, "{}└left:", prefix)?;
                node.print_tree_rec(f, level + 1)?;
            }
            None => writeln!(f, "{}└left: <None>", prefix)?,
        };

        match &self.righ {
            Some(node) => {
                writeln!(f, "{}└righ:", prefix)?;
                node.print_tree_rec(f, level + 1)?;
            }
            None => writeln!(f, "{}└righ: <None>", prefix)?,
        };

        Ok(())
    }
}

/// Tree
#[derive(Debug)]
pub struct Tree<T: TT> {
    root: Node<T>,
}

impl<T: TT> Tree<T> {
    /// new
    pub fn new(root_payload: T) -> Self {
        Self {
            root: Node::new(root_payload),
        }
    }

    /// insert
    pub fn insert(&mut self, index: &BitSlice<u8, Msb0>, payload: T) {
        self.root.insert_iter(index, payload)
    }

    /// in_order
    pub fn in_order(&self) -> String {
        self.root.print_in_order_rec()
    }

    /// pre_order
    pub fn pre_order(&self) -> String {
        self.root.print_pre_order_iter()
    }
}

struct FormatterTreeRecursive<'a, T: TT>(&'a Tree<T>);

impl<T: TT> Display for FormatterTreeRecursive<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "■")?;
        self.0.root.print_tree_rec(f, 0)
    }
}

struct FormatterTreeIterative<'a, T: TT>(&'a Tree<T>);

impl<T: TT> Display for FormatterTreeIterative<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "■")?;
        self.0.root.print_tree_iter(f)
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

impl TT for MyData {}

impl Display for MyData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.data2)
    }
}

#[cfg(test)]
mod test {
    use bitvec::{order::Msb0, view::BitView};

    use crate::bt::{FormatterTreeIterative, FormatterTreeRecursive, MyData, Tree};

    #[test]
    fn devel() {
        let root = MyData::new(99);
        let tree = Tree::new(root);
        let mut my = Vec::new();
        for i in 0..32 {
            my.push(MyData::new(i as u8));
        }

        println!("{}", FormatterTreeRecursive(&tree));
        println!("{}", FormatterTreeIterative(&tree));

        // let mut s;
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
        for i in 0..10 {
            let j: usize = (3 * i + 4) % 31;
            let j8 = j as u8;
            let index = j8.view_bits::<Msb0>();
            tree2.insert(index, my[j].clone());
            println!("j:{:3} bits:{}", j, index);
        }
        println!("{}", FormatterTreeRecursive(&tree2));
        println!("> {}", tree2.pre_order());

        let root = MyData::new(100);
        let mut tree3 = Tree::<MyData>::new(root);
        for i in 0..10 {
            let j: usize = (3 * i + 4) % 31;
            let j8 = j as u8;
            let index = j8.view_bits::<Msb0>();
            tree3.root.insert_iter(index, my[j].clone());
            println!("j:{:3} bits:{}", j, index);
        }
        println!("{}", FormatterTreeRecursive(&tree3));
        println!("{}", FormatterTreeIterative(&tree3));
        println!("> {}", tree3.root.print_pre_order_rec());
        println!("> {}", tree3.root.print_pre_order_iter());

        assert!(true);
    }
}
