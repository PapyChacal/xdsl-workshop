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
class TensorMakeOp(Operation):
    name = "riscv.toy.tensor.make"

    rd: Annotated[OpResult, RegisterType]
    rs1: Annotated[Operand, RegisterType]
    rs2: Annotated[Operand, RegisterType]
    rs3: Annotated[Operand, RegisterType]

    @classmethod
    def get(cls, shape_reg: Operation | SSAValue,
            data_reg: Operation | SSAValue,
            heap_reg: Operation | SSAValue) -> TensorMakeOp:
        return cls.build(operands=[shape_reg, data_reg, heap_reg],
                         result_types=[RegisterType()])


@irdl_op_definition
class TensorDataOp(Operation):
    name = "riscv.toy.tensor.data"

    rd: Annotated[OpResult, RegisterType]
    rs1: Annotated[Operand, RegisterType]

    @classmethod
    def get(cls, tensor_reg: Operation | SSAValue) -> TensorDataOp:
        return cls.build(operands=[tensor_reg], result_types=[RegisterType()])


@irdl_op_definition
class TensorShapeOp(Operation):
    name = "riscv.toy.tensor.shape"

    rd: Annotated[OpResult, RegisterType]
    rs1: Annotated[Operand, RegisterType]

    @classmethod
    def get(cls, tensor_reg: Operation | SSAValue) -> TensorShapeOp:
        return cls.build(operands=[tensor_reg], result_types=[RegisterType()])


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


@irdl_op_definition
class AllocOp(Operation):
    name = "riscv.toy.alloc"

    rd: Annotated[OpResult, RegisterType]
    rs1: Annotated[Operand, RegisterType]
    rs2: Annotated[Operand, RegisterType]

    @classmethod
    def get(cls, count_reg: Operation | SSAValue,
            heap_reg: Operation | SSAValue) -> AllocOp:
        return cls.build(operands=[count_reg, heap_reg],
                         result_types=[RegisterType()])


@irdl_op_definition
class BufferAddOp(Operation):
    name = "riscv.toy.buffer.add"

    rd: Annotated[OpResult, RegisterType]
    rs1: Annotated[Operand, RegisterType]
    rs2: Annotated[Operand, RegisterType]
    rs3: Annotated[Operand, RegisterType]
    rs4: Annotated[Operand, RegisterType]

    @classmethod
    def get(cls, count_reg: Operation | SSAValue, lhs: Operation | SSAValue,
            rhs: Operation | SSAValue,
            heap_reg: Operation | SSAValue) -> BufferAddOp:
        return cls.build(operands=[count_reg, lhs, rhs, heap_reg],
                         result_types=[RegisterType()])


ToyRISCV = Dialect([
    PrintTensorOp, AddTensorOp, TensorMakeOp, TensorDataOp, TensorShapeOp,
    ReshapeTensorOp, AllocOp, BufferAddOp
], [])
