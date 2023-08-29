use iced_x86::*;

use std::cmp::*;
use std::fmt;

use std::collections::BTreeMap;

use crate::binary::*;

// Basic Block: consecutive instructions up until the first jump
#[derive(Clone, Debug)]
pub struct BasicBlock {
    address: u64,
    instructions: Vec<Instruction>,
    targets: Vec<u64>,
}

// Ord and Eq traits for Basic Block struct
// Basic blocks are ordered acccording to their addresses
impl PartialEq for BasicBlock {
    fn eq(&self, other: &Self) -> bool {
        self.address == other.address
    }
}

impl Eq for BasicBlock {}

impl Ord for BasicBlock {
    fn cmp(&self, other: &Self) -> Ordering {
        self.address.cmp(&other.address)
    }
}

impl PartialOrd for BasicBlock {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
///////////////////////////////////////////////////////////////

impl BasicBlock {
    fn from_address(binary: &Binary, va: u64) -> Self {
        let mut bb: BasicBlock = BasicBlock {
            address: va,
            instructions: Vec::new(),
            targets: Vec::new(),
        };

        let byte_slice = binary.virtual_address_range(va..).unwrap();

        // set ip: given virtual address
        let mut decoder = Decoder::with_ip(64, byte_slice, va, 0);

        let mut instr = Instruction::default();

        decoder.decode_out(&mut instr);
        bb.instructions.push(instr);

        // TODO: deal with FlowControl::Call pattern !!

        loop {
            match instr.flow_control() {
                FlowControl::Next | FlowControl::Call => {
                    decoder.decode_out(&mut instr);
                    bb.instructions.push(instr);
                }
                FlowControl::ConditionalBranch => {
                    // is_jcc_short_or_near(), is_jcx_short(), is_loop(), is_loopcc()
                    // doesn ot exist: is_jkcc_short_or_near()
                    bb.targets.push(instr.next_ip());
                    bb.targets.push(instr.near_branch_target());
                    break;
                }
                FlowControl::UnconditionalBranch => {
                    if instr.is_jmp_short_or_near() {
                        bb.targets.push(instr.next_ip());
                        bb.targets.push(instr.near_branch_target());
                    } else if instr.is_jmp_far() {
                        bb.targets.push(instr.next_ip());
                        bb.targets.push(instr.far_branch_selector() as u64);
                    } else {
                        break;
                    }
                    break;
                }
                /* FlowControl::Call => {
                    if instr.is_call_near() {
                        bb.targets.push(instr.next_ip());
                        bb.targets.push(instr.near_branch_target());
                    } else if instr.is_call_far() {
                        bb.targets.push(instr.next_ip());
                        bb.targets.push(instr.far_branch_selector() as u64);
                    } else {
                        break;
                    }
                    break;
                } */
                FlowControl::Return
                | FlowControl::Interrupt
                | FlowControl::Exception
                | FlowControl::XbeginXabortXend
                | FlowControl::IndirectBranch
                | FlowControl::IndirectCall => {
                    break;
                }
            }
        }

        bb
    }

    // BasicBlock -> address of the last byte
    // maybe: address of the next instruction ??
    fn end_address(&self) -> u64 {
        let instr: Instruction = *(self.instructions).iter().last().unwrap();
        instr.next_ip() - 1
        // instr.ip() + (instr.len() as u64)
    }

    // BasicBlock + va -> address of the next valid instruction (if va = start then itself)
    fn next_valid_instr(&self, va: u64) -> Result<u64, String> {
        // TODO: what if it returns the next basic block's address ??
        let index = self
            .instructions
            .iter()
            .position(|x| x.ip() <= va && va < x.next_ip());
        match index {
            Some(i) => {
                if self.instructions[i].ip() == va {
                    Ok(va)
                } else {
                    Ok(self.instructions[i].next_ip())
                }
            }
            None => Err(String::from(
                "address is outside of basic block's range error",
            )),
        }
    }

    // BasicBlock + va -> cut the BB into two BBs at next_valid_instr(va)
    // the second block starts at next_valid_instr(va)
    fn cut_block(self, va: u64) -> Vec<BasicBlock> {
        let valid_va = self.next_valid_instr(va);
        match valid_va {
            Ok(addr) => {
                if self.address < addr && addr <= self.end_address() {
                    let cut_index = self
                        .instructions
                        .iter()
                        .position(|&x| x.ip() == addr)
                        .unwrap();
                    vec![
                        BasicBlock {
                            address: self.address,
                            instructions: self.instructions[..cut_index].to_vec(),
                            targets: vec![addr],
                        },
                        BasicBlock {
                            address: addr,
                            instructions: self.instructions[cut_index..].to_vec(),
                            targets: self.targets,
                        },
                    ]
                } else {
                    vec![self]
                }
            }
            Err(_) => {
                vec![self]
            }
        }
    }

    // BasicBlock -> address (u64)
    pub fn address(&self) -> u64 {
        self.address
    }

    // BasicBlock -> targets (&[u64])
    pub fn targets(&self) -> &[u64] {
        &self.targets
    }

    // BasicBlock -> instructions (&[Instruction])
    pub fn instructions(&self) -> &[Instruction] {
        &self.instructions
    }
}

impl fmt::Display for BasicBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "address:\n       {:016x}\n", &self.address)?;
        writeln!(f, "basic block:")?;

        // options format:
        // https://docs.rs/iced-x86/latest/src/iced_x86/instruction.rs.html#3767-3797
        let mut formatter = MasmFormatter::new();
        formatter.options_mut().set_branch_leading_zeros(false);
        formatter.options_mut().set_uppercase_hex(false);

        for instruction in &self.instructions {
            write!(f, "      {:016x}: ", instruction.ip())?;

            let mut output = String::new();
            formatter.format(instruction, &mut output);
            f.write_str(&output)?;

            writeln!(f)?;
        }

        writeln!(f, "target(s):")?;

        for element in &self.targets {
            writeln!(f, "      {:016x}", element)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // TEST: next_valid_instr() method
    #[test]
    fn next_valid_va() {
        let path = String::from("/home/san-rok/projects/testtest/target/debug/testtest");
        let binary = Binary::from_elf(path);

        let virtual_address: u64 = 0x8840;

        let bb = BasicBlock::from_address(&binary, virtual_address);

        assert_eq!(
            Err(String::from(
                "address is outside of basic block's range error"
            )),
            bb.next_valid_instr(0x8838)
        );
        assert_eq!(
            Err(String::from(
                "address is outside of basic block's range error"
            )),
            bb.next_valid_instr(0x8853)
        );

        assert_eq!(Ok(0x8847), bb.next_valid_instr(0x8842));
        assert_eq!(Ok(0x8847), bb.next_valid_instr(0x8846));
        assert_eq!(Ok(0x884e), bb.next_valid_instr(0x8849));
        assert_eq!(Ok(0x884e), bb.next_valid_instr(0x884e));
        assert_eq!(Ok(0x8853), bb.next_valid_instr(0x8852));
    }
}

// Control Flow Graph: graph of basic blocks
pub struct ControlFlowGraph {
    address: u64,
    blocks: Vec<BasicBlock>,
}

impl ControlFlowGraph {
    // explore control flow graph from a given virtual address (using DFS)
    pub fn from_address(binary: &Binary, va: u64) -> Self {
        let mut blocks: BTreeMap<u64, BasicBlock> = BTreeMap::new();
        let mut addresses: Vec<u64> = Vec::new();

        addresses.push(va);

        while let Some(address) = addresses.pop() {
            let bb = BasicBlock::from_address(binary, address);

            // is this clone too much?
            let mut targets = bb.targets().to_vec();

            blocks.insert(bb.address(), bb);

            while let Some(target) = targets.pop() {
                let cut = blocks.range(..target).next_back().map(|(&x, _)| x);

                match cut {
                    Some(addr) if target <= blocks.get(&addr).unwrap().end_address() => {
                        let tmp_block = blocks.remove(&addr).unwrap();
                        let cut_blocks = tmp_block.cut_block(target);
                        for i in cut_blocks {
                            blocks.insert(i.address(), i);
                        }
                    }
                    _ => {
                        if !addresses.contains(&target) {
                            addresses.push(target);
                        }
                    }
                }
            }
        }

        let mut blocks: Vec<BasicBlock> = blocks.into_values().collect::<Vec<BasicBlock>>();
        blocks.sort();

        ControlFlowGraph {
            address: va,
            blocks,
        }
    }

    // Graph -> address (u64)
    pub fn address(&self) -> u64 {
        self.address
    }

    // Graph -> blocks (&[BasicBlock])
    pub fn blocks(&self) -> &[BasicBlock] {
        &self.blocks
    }

    // from graph to .dot
    pub fn render_to<W: std::io::Write>(&self, output: &mut W) -> dot2::Result {
        dot2::render(self, output)
    }
}

impl<'a> dot2::Labeller<'a> for ControlFlowGraph {
    type Node = u64;
    type Edge = (u64, u64);
    type Subgraph = ();

    // .dot compatible identifier naming the graph
    fn graph_id(&'a self) -> dot2::Result<dot2::Id<'a>> {
        dot2::Id::new("control_flow")
    }

    // maps n to unique (valid .dot) identifier
    fn node_id(&'a self, n: &Self::Node) -> dot2::Result<dot2::Id<'a>> {
        dot2::Id::new(format!("N0x{:x}", n))
    }

    // labels of nodes
    fn node_label(&'a self, n: &Self::Node) -> dot2::Result<dot2::label::Text<'a>> {
        let label = self
            .blocks
            .iter()
            .find(|&v| v.address() == *n)
            .map(|v| format!("{}", v))
            .unwrap();

        Ok(dot2::label::Text::LabelStr(label.into()))
    }
}

impl<'a> dot2::GraphWalk<'a> for ControlFlowGraph {
    type Node = u64;
    type Edge = (u64, u64);
    type Subgraph = ();

    // all nodes of the graph
    fn nodes(&self) -> dot2::Nodes<'a, Self::Node> {
        self.blocks().iter().map(|n| n.address()).collect()
    }

    // all edges of the graph
    fn edges(&'a self) -> dot2::Edges<'a, Self::Edge> {
        let mut edges: Vec<(u64, u64)> = Vec::new();

        for block in self.blocks() {
            let source = block.address();
            for target in block.targets() {
                edges.push((source, *target));
            }
        }

        edges.into_iter().collect()
    }

    // source node for the given edge
    fn source(&self, edge: &Self::Edge) -> Self::Node {
        let &(s, _) = edge;
        s
    }

    // target node for the given edge
    fn target(&self, edge: &Self::Edge) -> Self::Node {
        let &(_, t) = edge;
        t
    }
}
