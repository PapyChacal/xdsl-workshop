from xdsl.dialects.builtin import i32, ModuleOp, UnrankedTensorType
from xdsl.ir import BlockArgument, Operation, OpResult, SSAValue
from xdsl.printer import Printer

from ..dialect import (ConstantOp, FuncOp, GenericCallOp, MulOp, ReturnOp,
                       ReshapeOp, TransposeOp)
import toy.dialect as toy

from ..building import Builder, build_callable

from io import StringIO


def op_desc(op: Operation) -> str:
    stream = StringIO()
    Printer(stream=stream, target=Printer.Target.MLIR).print(op)
    return stream.getvalue()


def new_module() -> ModuleOp:

    unrankedTensorTypeI32 = UnrankedTensorType.from_type(i32)

    @toy.func_op('multiply_transpose', private=True)
    @build_callable([unrankedTensorTypeI32, unrankedTensorTypeI32],
                    [unrankedTensorTypeI32])
    def multiply_transpose(builder: Builder, arg0: SSAValue,
                           arg1: SSAValue) -> None:
        a_t = toy.transpose(builder, arg0)
        b_t = toy.transpose(builder, arg1)
        prod = toy.mul(builder, a_t, b_t)
        toy.return_(builder, prod)

    def call_multiply_transpose(builder: Builder, a: SSAValue,
                                b: SSAValue) -> OpResult:
        return toy.generic_call(builder, 'multiply_transpose', [a, b],
                                [unrankedTensorTypeI32])

    @toy.func_op('main')
    @build_callable([], [])
    def main(builder: Builder) -> None:
        a = toy.constant(builder, [1, 2, 3, 4, 5, 6], [2, 3])
        b_0 = toy.constant(builder, [1, 2, 3, 4, 5, 6], [6])
        b = toy.reshape(builder, b_0, [2, 3])
        c = call_multiply_transpose(builder, a, b)
        call_multiply_transpose(builder, b, a)
        call_multiply_transpose(builder, b, c)
        a_t = toy.transpose(builder, a)
        call_multiply_transpose(builder, a_t, c)
        toy.return_(builder)

    module = ModuleOp.from_region_or_ops([
        multiply_transpose,
        main,
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
