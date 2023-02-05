from xdsl.ir import MLContext, Operation
from xdsl.dialects.builtin import ModuleOp
from xdsl.printer import Printer

from toy.dialect import Toy

from riscv.riscv_ssa import LabelOp, LIOp, MULOp, AddOp, ECALLOp, RISCVSSA, DirectiveOp, LWOp

context = MLContext()

context.register_dialect(Toy)
context.register_dialect(RISCVSSA)

printer = Printer(target=Printer.Target.MLIR)

example = """
def main() {
  var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];
  var c = a + b;
  print(c);
}
"""

# bla: tuple[Operation, ...] = (
#     DirectiveOp.get(".bss", ""),  # bss is standard name for heap
#     LabelOp.get("heap"),
#     DirectiveOp.get(".space", "16k"),
#     DirectiveOp.get(".data", ""),
#     LabelOp.get("main.a.data"),
# )

# risc_v_module = ModuleOp.from_region_or_ops(
#     list((
#         DirectiveOp.get(".words", "0x6, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6"),
#         LabelOp.get("main.a.shape"),
#         DirectiveOp.get(".words", "0x2, 0x2, 0x3"),
#         LabelOp.get("main.b.data"),
#         DirectiveOp.get(".words", "0x6, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6"),
#         LabelOp.get("main.b.shape"),
#         DirectiveOp.get(".text", ""),
#         LabelOp.get('malloc'),  # how to do arguments?
#         malloc_size := LIOp.get("arg0"),
#         LabelOp.get('main'),
#         a_shape := LIOp.get("main.a.shape"),
#         a_data := LIOp.get("main.a.data"),
#         a_rank := LWOp.get(a, 0),
#         a_count_ptr := AddOp.get(a, a_rank),
#         a_count := LWOp.get(a, 0),
#         a0 := LIOp.get(82),
#         a1 := LIOp.get(5),
#         mul := MULOp.get(a0, a1),
#         a2 := LIOp.get(10),
#         add := AddOp.get(mul, a2),
#         ECALLOp.get(93, add))))

# printer.print(risc_v_module)