from riscv.riscv_ssa import *
from xdsl.dialects.builtin import ModuleOp

module = ModuleOp.from_region_or_ops([
    SectionOp.from_ops('.text', [
        FuncOp.from_ops('main', [
            a0 := LIOp.get(83),
            a1 := LIOp.get(5),
            mul := MULOp.get(a0, a1),
            a2 := LIOp.get(10),
            add := AddOp.get(mul, a2),
            PrintOp.get(add),  # debug instruction to print register contents
            ReturnOp.get(),
        ])
    ])
])

printer = Printer(target=Printer.Target.MLIR)
