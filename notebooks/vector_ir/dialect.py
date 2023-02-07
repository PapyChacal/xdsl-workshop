from __future__ import annotations

from xdsl.ir import Operation, SSAValue, Dialect, OpResult
from xdsl.irdl import irdl_op_definition, Operand, OpAttr
from xdsl.dialects.builtin import IntAttr, ArrayAttr, StringAttr

from typing import Annotated

from toy.dialect import NoSideEffect

from riscv.riscv_ssa import RegisterType


@irdl_op_definition
class VectorAddOp(Operation, NoSideEffect):
    name = "toy.vector.add"

    rd: Annotated[OpResult, RegisterType]
    rs1: Annotated[Operand, RegisterType]
    rs2: Annotated[Operand, RegisterType]

    @classmethod
    def get(cls, lhs_reg: Operation | SSAValue,
            rhs_reg: Operation | SSAValue) -> VectorAddOp:
        return cls.build(operands=[lhs_reg, rhs_reg],
                         result_types=[RegisterType()])


@irdl_op_definition
class VectorConstantOp(Operation, NoSideEffect):
    """
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = riscv.toy.vector_constant array<[1, 2, 3, 4, 5, 6]>: array<i32>
    ```
    """
    name: str = "toy.vector.constant"
    data: OpAttr[ArrayAttr[IntAttr]]
    label: OpAttr[StringAttr]
    res: Annotated[OpResult, RegisterType]

    @staticmethod
    def get(data: list[int] | ArrayAttr[IntAttr],
            label: str | StringAttr) -> VectorConstantOp:
        if isinstance(data, list):
            data = ArrayAttr.from_list([IntAttr.from_int(i) for i in data])
        if isinstance(label, str):
            label = StringAttr.from_str(label)
        return VectorConstantOp.create(result_types=[RegisterType()],
                                       attributes={
                                           "data": data,
                                           "label": label
                                       })

    def get_data(self) -> list[int]:
        return [el.data for el in self.data.data]


@irdl_op_definition
class TensorMakeOp(Operation, NoSideEffect):
    name = "toy.tensor.make"

    rd: Annotated[OpResult, RegisterType]
    rs1: Annotated[Operand, RegisterType]
    rs2: Annotated[Operand, RegisterType]

    @classmethod
    def get(cls, shape_reg: Operation | SSAValue,
            data_reg: Operation | SSAValue) -> TensorMakeOp:
        return cls.build(operands=[shape_reg, data_reg],
                         result_types=[RegisterType()])


@irdl_op_definition
class TensorDataOp(Operation, NoSideEffect):
    name = "toy.tensor.data"

    rd: Annotated[OpResult, RegisterType]
    rs1: Annotated[Operand, RegisterType]

    @classmethod
    def get(cls, tensor_reg: Operation | SSAValue) -> TensorDataOp:
        return cls.build(operands=[tensor_reg], result_types=[RegisterType()])


@irdl_op_definition
class TensorShapeOp(Operation, NoSideEffect):
    name = "toy.tensor.shape"

    rd: Annotated[OpResult, RegisterType]
    rs1: Annotated[Operand, RegisterType]

    @classmethod
    def get(cls, tensor_reg: Operation | SSAValue) -> TensorShapeOp:
        return cls.build(operands=[tensor_reg], result_types=[RegisterType()])


ToyRISCV = Dialect([
    VectorAddOp,
    TensorMakeOp,
    TensorDataOp,
    TensorShapeOp,
], [])
