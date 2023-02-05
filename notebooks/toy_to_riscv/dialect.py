from __future__ import annotations

from xdsl.ir import Operation, SSAValue, Dialect, OpResult
from xdsl.irdl import irdl_op_definition, Operand

from typing import Annotated

from riscv.riscv_ssa import RegisterType


@irdl_op_definition
class PrintTensorOp(Operation):
    name = "riscv.toy.print"

    rs1: Annotated[Operand, RegisterType]

    @classmethod
    def get(cls, rs1: Operation | SSAValue) -> PrintTensorOp:
        return cls.build(operands=[rs1], result_types=[])


@irdl_op_definition
class AddTensorOp(Operation):
    name = "riscv.toy.add"

    rd: Annotated[OpResult, RegisterType]
    rs1: Annotated[Operand, RegisterType]
    rs2: Annotated[Operand, RegisterType]
    rs3: Annotated[Operand, RegisterType]

    @classmethod
    def get(cls, lhs_reg: Operation | SSAValue, rhs_reg: Operation | SSAValue,
            heap_reg: Operation | SSAValue) -> AddTensorOp:
        return cls.build(operands=[lhs_reg, rhs_reg, heap_reg],
                         result_types=[RegisterType()])


@irdl_op_definition
class ReshapeTensorOp(Operation):
    name = "riscv.toy.reshape"

    rd: Annotated[OpResult, RegisterType]
    rs1: Annotated[Operand, RegisterType]
    rs2: Annotated[Operand, RegisterType]
    rs3: Annotated[Operand, RegisterType]

    @classmethod
    def get(cls, input_reg: Operation | SSAValue,
            shape_reg: Operation | SSAValue,
            heap_reg: Operation | SSAValue) -> ReshapeTensorOp:
        return cls.build(operands=[input_reg, shape_reg, heap_reg],
                         result_types=[RegisterType()])


ToyRISCV = Dialect([PrintTensorOp, AddTensorOp, ReshapeTensorOp], [])
