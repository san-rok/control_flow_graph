use control_flow_graph::*;

fn main() {
    let path = String::from("/home/san-rok/projects/testtest/target/debug/testtest");
    let binary = Binary::from_elf(path);

    let virtual_address: u64 = 0x8840;
    // test: 0x88cb, 0x8870, 0x88b0, 0x8a0d, 0x893e, 0x88f0, 0x8c81, 0x8840, 0x8f41, 0x970b, 0x96b4

    // let cfg: ControlFlowGraph = ControlFlowGraph::from_address(&binary, virtual_address);
    let cfg: ControlFlowGraph = ControlFlowGraph::parallel_from_address(&binary, virtual_address);

    let mut f =
        std::fs::File::create("/home/san-rok/projects/control_flow_graph/cfgraph.dot").unwrap();
    cfg.render_to(&mut f).unwrap();
    // dot -Tsvg cfgraph.dot > cfgraph.svg
}
