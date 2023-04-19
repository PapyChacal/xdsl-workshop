from io import StringIO
from compiler import compile
from riscv.emulator_iop import run_riscv
from riscv_buffer_ir.accelerator import ToyAccelerator

program = """
def main() {
  # Define a variable `a` with shape <2, 3>, initialized with the literal value.
  # The shape is inferred from the supplied literal.
  var a = [[1, 2, 3], [4, 5, 6]];

  # b is identical to a, the literal tensor is implicitly reshaped: defining new
  # variables is the way to reshape tensors (element count must match).
  var b<3, 2> = [1, 2, 3, 4, 5, 6];

  # There is a built-in print instruction to display the contents of the tensor
  print(b);

  # Reshapes are implicit on assignment
  var c<2, 3> = b;

  # There are + and * operators for pointwise addition and multiplication
  var d = a + c;

  print(d);
}
"""


def test_execution():
    code = compile(program)
    stream = StringIO()
    ToyAccelerator.stream = stream
    run_riscv(code,
              extensions=[ToyAccelerator],
              unlimited_regs=True,
              verbosity=1)
    assert '[[2, 4, 6], [8, 10, 12]]' in stream.getvalue()
