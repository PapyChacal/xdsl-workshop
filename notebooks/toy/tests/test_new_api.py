from dataclasses import dataclass, field

from xdsl.dialects.builtin import i32, ModuleOp, UnrankedTensorType
from xdsl.ir import BlockArgument, Operation, OpResult
from xdsl.printer import Printer

from ..dialect import (ConstantOp, FuncOp, GenericCallOp, MulOp, ReturnOp,
                       ReshapeOp, TransposeOp)

from io import StringIO


def op_desc(op: Operation) -> str:
    stream = StringIO()
    Printer(stream=stream, target=Printer.Target.MLIR).print(op)
    return stream.getvalue()


@dataclass
class OpListBuilder:
    ops: list[Operation] = field(default_factory=list)

    def get_ops(self) -> list[Operation]:
        return self.ops

    def add_op(self, op: Operation):
        self.ops.append(op)


def foo_build(builder: OpListBuilder, op: Operation) -> tuple[OpResult, ...]:
    builder.add_op(op)
    return tuple(op.results)


def new_module() -> ModuleOp:

    def multiply_transpose_def(*args: BlockArgument) -> list[Operation]:
        arg0, arg1 = args
        builder = OpListBuilder()

        a_t, = foo_build(builder, TransposeOp.from_input(arg0))
        b_t, = foo_build(builder, TransposeOp.from_input(arg1))
        prod, = foo_build(builder, MulOp.from_summands(a_t, b_t))
        foo_build(builder, ReturnOp.from_input(prod))

        return builder.get_ops()

    def main_def(*args: BlockArgument) -> list[Operation]:
        builder = OpListBuilder()

        a, = foo_build(
            builder,
            ConstantOp.from_list([1, 2, 3, 4, 5, 6], [2, 3]),
        )
        b_0, = foo_build(
            builder,
            ConstantOp.from_list([1, 2, 3, 4, 5, 6], [6]),
        )
        b, = foo_build(
            builder,
            ReshapeOp.from_input(b_0, [2, 3]),
        )
        c, = foo_build(
            builder,
            GenericCallOp.get('multiply_transpose', [a, b],
                              [unrankedTensorTypeI32]),
        )
        foo_build(
            builder,
            GenericCallOp.get('multiply_transpose', [b, a],
                              [unrankedTensorTypeI32]),
        )
        foo_build(
            builder,
            GenericCallOp.get('multiply_transpose', [b, c],
                              [unrankedTensorTypeI32]),
        )
        a_t, = foo_build(
            builder,
            TransposeOp.from_input(a),
        )
        foo_build(
            builder,
            GenericCallOp.get('multiply_transpose', [a_t, c],
                              [unrankedTensorTypeI32]),
        )
        foo_build(builder, ReturnOp.from_input())

        return builder.get_ops()

    unrankedTensorTypeI32 = UnrankedTensorType.from_type(i32)

    module = ModuleOp.from_region_or_ops([
        FuncOp.from_callable('multiply_transpose',
                             [unrankedTensorTypeI32, unrankedTensorTypeI32],
                             [unrankedTensorTypeI32],
                             multiply_transpose_def,
                             private=True),
        FuncOp.from_callable('main', [], [], main_def),
    ])

    return module


def test_convert_ast():
    ref_op = old_module()
    new_op = new_module()

    ref_desc = op_desc(ref_op)
    new_desc = op_desc(new_op)

    ref_lines = ref_desc.split('\n')
    new_lines = new_desc.split('\n')

    for i, (l, r) in enumerate(zip(ref_lines, new_lines)):
        assert l == r, '\n'.join(new_lines[:i])

    assert ref_op.is_structurally_equivalent(new_op)


def old_module() -> ModuleOp:
    unrankedi32TensorType = UnrankedTensorType.from_type(i32)

    def func_body(*args: BlockArgument) -> list[Operation]:
        arg0, arg1 = args
        f0 = TransposeOp.from_input(arg0)
        f1 = TransposeOp.from_input(arg1)
        f2 = MulOp.from_summands(f0.results[0], f1.results[0])
        f3 = ReturnOp.from_input(f2.results[0])
        return [f0, f1, f2, f3]

    def main_body(*args: BlockArgument) -> list[Operation]:
        m0 = ConstantOp.from_list([1, 2, 3, 4, 5, 6], [2, 3])
        [a] = m0.results
        m1 = ConstantOp.from_list([1, 2, 3, 4, 5, 6], [6])
        m2 = ReshapeOp.from_input(m1.results[0], [2, 3])
        [b] = m2.results
        m3 = GenericCallOp.get('multiply_transpose', [a, b],
                               [unrankedi32TensorType])
        [c] = m3.results
        m4 = GenericCallOp.get('multiply_transpose', [b, a],
                               [unrankedi32TensorType])
        m5 = GenericCallOp.get('multiply_transpose', [b, c],
                               [unrankedi32TensorType])
        m6 = TransposeOp.from_input(a)
        [a_transposed] = m6.results
        m7 = GenericCallOp.get('multiply_transpose', [a_transposed, c],
                               [unrankedi32TensorType])
        m8 = ReturnOp.from_input()
        return [m0, m1, m2, m3, m4, m5, m6, m7, m8]

    multiply_transpose = FuncOp.from_callable(
        'multiply_transpose', [unrankedi32TensorType, unrankedi32TensorType],
        [unrankedi32TensorType],
        func_body,
        private=True)
    main = FuncOp.from_callable('main', [], [], main_body, private=False)

    module_op = ModuleOp.from_region_or_ops([multiply_transpose, main])

    return module_op
