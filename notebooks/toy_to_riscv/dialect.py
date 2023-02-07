from __future__ import annotations

from xdsl.ir import Operation, SSAValue, Dialect, OpResult
from xdsl.irdl import irdl_op_definition, Operand

from typing import Annotated

from toy.dialect import NoSideEffect

from riscv.riscv_ssa import RegisterType


@irdl_op_definition
class TensorPrintOp(Operation):
    name = "riscv.toy.print"

    rs1: Annotated[Operand, RegisterType]

    @classmethod
    def get(cls, rs1: Operation | SSAValue) -> TensorPrintOp:
        return cls.build(operands=[rs1], result_types=[])


@irdl_op_definition
class AllocOp(Operation, NoSideEffect):
    """
    Allocate a buffer of `count` ints, or `count` * 4 bytes
    """
    name = "riscv.buffer.alloc"

    rd: Annotated[OpResult, RegisterType]
    rs1: Annotated[Operand, RegisterType]

    @classmethod
    def get(cls, count_reg: Operation | SSAValue) -> AllocOp:
        return cls.build(operands=[count_reg], result_types=[RegisterType()])


@irdl_op_definition
class BufferAddOp(Operation, NoSideEffect):
    name = "riscv.buffer.add"

    rs1: Annotated[Operand, RegisterType]
    rs2: Annotated[Operand, RegisterType]
    rs3: Annotated[Operand, RegisterType]

    @classmethod
    def get(cls, count_reg: Operation | SSAValue, source: Operation | SSAValue,
            destination: Operation | SSAValue) -> BufferAddOp:
        return cls.build(operands=[count_reg, source, destination])


ToyRISCV = Dialect([
    TensorPrintOp,
    AllocOp,
    BufferAddOp,
], [])
