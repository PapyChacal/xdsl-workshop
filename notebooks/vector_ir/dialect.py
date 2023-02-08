from __future__ import annotations

from typing import TypeAlias

from xdsl.ir import Operation, SSAValue, Dialect, OpResult
from xdsl.irdl import irdl_op_definition, Operand, OpAttr
from xdsl.dialects.builtin import StringAttr, VectorType, IntegerType, DenseIntOrFPElementsAttr, i32

from typing import Annotated

from riscv.riscv_ssa import RegisterType

VectorTypeI32: TypeAlias = VectorType[IntegerType]


class NoSideEffect:
    pass


@irdl_op_definition
class VectorAddOp(Operation, NoSideEffect):
    name = "vector.add"

    res: Annotated[OpResult, VectorTypeI32]
    lhs: Annotated[Operand, VectorTypeI32]
    rhs: Annotated[Operand, VectorTypeI32]

    @classmethod
    def get(cls, lhs: Operation | SSAValue,
            rhs: Operation | SSAValue) -> VectorAddOp:
        if isinstance(lhs, Operation):
            lhs = lhs.results[0]
        return cls.build(operands=[lhs, rhs], result_types=[lhs.typ])


@irdl_op_definition
class VectorConstantOp(Operation, NoSideEffect):
    """
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = riscv.vector_constant array<[1, 2, 3, 4, 5, 6]>: array<i32>
    ```
    """
    name: str = "vector.constant"
    data: OpAttr[DenseIntOrFPElementsAttr]
    label: OpAttr[StringAttr]
    res: Annotated[OpResult, VectorTypeI32]

    @staticmethod
    def get(data: list[int] | DenseIntOrFPElementsAttr,
            label: str | StringAttr) -> VectorConstantOp:
        if isinstance(data, list):
            data = DenseIntOrFPElementsAttr.vector_from_list(data, i32)
        if isinstance(label, str):
            label = StringAttr.from_str(label)
        result_type = data.type
        return VectorConstantOp.create(result_types=[result_type],
                                       attributes={
                                           "data": data,
                                           "label": label
                                       })

    def get_data(self) -> list[int]:
        return [int(el.value.data) for el in self.data.data.data]


ToyRISCV = Dialect([
    VectorAddOp,
], [])
