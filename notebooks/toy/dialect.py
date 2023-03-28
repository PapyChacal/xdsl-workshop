"""
Toy language dialect from MLIR tutorial.
"""

from __future__ import annotations

from typing import Annotated, List, TypeAlias, Union, Optional, Any, cast
from functools import reduce

from xdsl.ir import (Dialect, SSAValue, Attribute, Block, Region, Operation,
                     OpResult)
from xdsl.dialects.builtin import (IntegerType, FunctionType,
                                   FlatSymbolRefAttr, TensorType,
                                   UnrankedTensorType, i32,
                                   DenseIntOrFPElementsAttr, StringAttr)
from xdsl.irdl import (
    OpAttr,
    Operand,
    OptOpAttr,
    OptOperand,
    VarOpResult,
    VarOperand,
    irdl_op_definition,
    AnyAttr,
)

from xdsl.utils.exceptions import VerifyException

import vector_ir.dialect as vd

from vector_ir.dialect import NoSideEffect

from .building import Builder

TensorTypeI32: TypeAlias = TensorType[IntegerType]
UnrankedTensorTypeI32: TypeAlias = UnrankedTensorType[IntegerType]
AnyTensorTypeI32: TypeAlias = TensorTypeI32 | UnrankedTensorTypeI32


@irdl_op_definition
class ConstantOp(Operation, NoSideEffect):
    """
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = toy.constant dense<[[1, 2, 3], [4, 5, 6]]>
                        : tensor<2x3xi32>
    ```
    """
    name: str = "toy.constant"
    value: OpAttr[DenseIntOrFPElementsAttr]
    res: Annotated[OpResult, TensorTypeI32]

    @staticmethod
    def from_list(data: List[int], shape: List[int]) -> ConstantOp:
        value = DenseIntOrFPElementsAttr.tensor_from_list(data, i32, shape)
        return ConstantOp.from_value(value)

    @staticmethod
    def from_value(value: DenseIntOrFPElementsAttr) -> ConstantOp:
        return ConstantOp.create(result_types=[value.type],
                                 attributes={"value": value})

    def verify_(self) -> None:
        if not self.res.typ == self.value.type:
            raise VerifyException(
                "Expected value and result types to be equal: "
                f"{self.res.typ}, {self.value.type}")

    def get_type(self) -> TensorTypeI32:
        # Constant cannot be unranked
        return cast(TensorTypeI32, self.value.type)

    def get_shape(self) -> list[int]:
        return self.get_type().get_shape()

    def get_data(self) -> list[int]:
        return [int(el.value.data) for el in self.value.data.data]


@irdl_op_definition
class AddOp(Operation, NoSideEffect):
    """
    The "add" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
    """
    name: str = 'toy.add'
    lhs: Annotated[Operand, AnyTensorTypeI32]
    rhs: Annotated[Operand, AnyTensorTypeI32]
    res: Annotated[OpResult, AnyTensorTypeI32]

    @classmethod
    def from_summands(cls: type[AddOp], lhs: SSAValue, rhs: SSAValue) -> AddOp:
        assert isinstance(lhs.typ, TensorType | UnrankedTensorType)
        if isinstance(lhs.typ, TensorType):
            result_typ = cast(TensorType[Any], lhs.typ)
        else:
            result_typ = rhs.typ
        return cls.create(result_types=[result_typ], operands=[lhs, rhs])

    def verify_(self):
        args = [self.lhs, self.rhs]

        shape = None
        for arg in args:
            # Expect shapes to be the same whenever they are defined, no check for unranked
            if isinstance(arg.typ, TensorType):
                if shape is None:
                    shape = arg.typ.shape
                else:
                    if shape != arg.typ.shape:
                        raise VerifyException(
                            "Expected AddOp args to have the same shape")


@irdl_op_definition
class FuncOp(Operation):
    """
    The "toy.func" operation represents a user defined function. These are
    callable SSA-region operations that contain toy computations.

    Example:

    ```mlir
    toy.func @main() {
      %0 = toy.constant dense<5.500000e+00> : tensor<i32>
      %1 = toy.reshape(%0 : tensor<i32>) to tensor<2x2xi32>
      toy.print %1 : tensor<2x2xi32>
      toy.return
    }
    ```
    """
    name = 'toy.func'
    body: Region
    sym_name: OpAttr[StringAttr]
    function_type: OpAttr[FunctionType]
    sym_visibility: OptOpAttr[StringAttr]

    @staticmethod
    def from_region(name: str,
                    ftype: FunctionType,
                    region: Region,
                    /,
                    private: bool = False):
        attributes: dict[str, Attribute] = {
            "sym_name": StringAttr.from_str(name),
            "function_type": ftype,
        }
        if private:
            attributes["sym_visibility"] = StringAttr.from_str("private")

        return FuncOp.create(attributes=attributes, regions=[region])

    @staticmethod
    def from_callable(name: str,
                      input_types: List[Attribute],
                      return_types: List[Attribute],
                      func: Block.BlockCallback,
                      /,
                      private: bool = False):
        ftype = FunctionType.from_lists(input_types, return_types)
        return FuncOp.from_region(
            name,
            ftype,
            Region.from_block_list([Block.from_callable(input_types, func)]),
            private=private)

    @staticmethod
    def from_callable_region(name: str,
                             callable_region: tuple[Region, FunctionType],
                             /,
                             private: bool = False) -> FuncOp:
        region, ftype = callable_region
        return FuncOp.from_region(name, ftype, region, private=private)

    def verify_(self):
        # Check that the returned value matches the type of the function
        if len(self.body.blocks) != 1:
            raise VerifyException("Expected FuncOp to contain one block")

        block = self.body.blocks[0]

        if not len(block.ops):
            raise VerifyException("Expected FuncOp to not be empty")

        last_op = block.ops[-1]

        if not isinstance(last_op, ReturnOp):
            raise VerifyException(
                "Expected last op of FuncOp to be a ReturnOp")

        operand = last_op.input
        operand_typ = None if operand is None else operand.typ

        return_typs = self.function_type.outputs.data

        if len(return_typs):
            if len(return_typs) == 1:
                return_typ = return_typs[0]
            else:
                raise VerifyException(
                    "Expected return type of func to have 0 or 1 values")
        else:
            return_typ = None

        if operand_typ != return_typ:
            raise VerifyException(
                "Expected return value to match return type of function")


@irdl_op_definition
class GenericCallOp(Operation):
    name: str = "toy.generic_call"
    arguments: Annotated[VarOperand, AnyAttr()]
    callee: OpAttr[FlatSymbolRefAttr]

    # Note: naming this results triggers an ArgumentError
    res: Annotated[VarOpResult, AnyTensorTypeI32]

    @classmethod
    def get(cls: type[GenericCallOp], callee: Union[str, FlatSymbolRefAttr],
            operands: List[Union[SSAValue, OpResult]],
            return_types: List[Attribute]) -> GenericCallOp:
        if isinstance(callee, str):
            callee = FlatSymbolRefAttr.from_str(callee)

        return cls.create(operands=operands,
                          result_types=return_types,
                          attributes={"callee": callee})


@irdl_op_definition
class MulOp(Operation, NoSideEffect):
    """
    The "mul" operation performs element-wise multiplication between two
    tensors. The shapes of the tensor operands are expected to match.
    """
    name: str = 'toy.mul'
    lhs: Annotated[Operand, AnyTensorTypeI32]
    rhs: Annotated[Operand, AnyTensorTypeI32]
    res: Annotated[OpResult, AnyTensorTypeI32]

    @classmethod
    def from_summands(cls: type[MulOp], lhs: SSAValue, rhs: SSAValue) -> MulOp:
        if isinstance(lhs.typ, TensorType):
            result_typ = cast(TensorType[Any], lhs.typ)
        else:
            result_typ = rhs.typ
        return cls.create(result_types=[result_typ], operands=[lhs, rhs])

    def verify_(self):
        args = [self.lhs, self.rhs]

        shape = None
        for arg in args:
            # Expect shapes to be the same whenever they are defined, no check for unranked
            if isinstance(arg.typ, TensorType):
                if shape is None:
                    shape = arg.typ.shape
                else:
                    if shape != arg.typ.shape:
                        raise VerifyException(
                            "Expected MulOp args to have the same shape")


@irdl_op_definition
class PrintOp(Operation):
    """
    The "print" builtin operation prints a given input tensor, and produces
    no results.
    """
    name: str = 'toy.print'
    input: Annotated[Operand, AnyAttr()]

    @classmethod
    def from_input(cls: type[PrintOp], input: SSAValue) -> PrintOp:
        return cls.create(operands=[input])


@irdl_op_definition
class ReturnOp(Operation):
    """
    The "return" operation represents a return operation within a function.
    The operation takes an optional tensor operand and produces no results.
    The operand type must match the signature of the function that contains
    the operation. For example:

    ```mlir
      func @foo() -> tensor<2xi32> {
        ...
        toy.return %0 : tensor<2xi32>
      }
    ```
    """
    name: str = 'toy.return'
    input: Annotated[OptOperand, AnyTensorTypeI32]

    @classmethod
    def from_input(cls: type[ReturnOp],
                   input: Optional[SSAValue] = None) -> ReturnOp:
        return cls.create(operands=[input] if input is not None else [])


@irdl_op_definition
class ReshapeOp(Operation, NoSideEffect):
    """
    Reshape operation is transforming its input tensor into a new tensor with
    the same number of elements but different shapes. For example:

    ```mlir
       %0 = toy.reshape (%arg1 : tensor<10xi32>) to tensor<5x2xi32>
    ```
    """
    name: str = 'toy.reshape'
    arg: Annotated[Operand, AnyTensorTypeI32]
    # We expect that the reshape operation returns a statically shaped tensor.
    res: Annotated[OpResult, TensorTypeI32]

    @classmethod
    def from_input(cls: type[ReshapeOp], arg: SSAValue,
                   shape: List[int]) -> ReshapeOp:
        assert isinstance(arg.typ, TensorType | UnrankedTensorType)
        element_type = cast(IntegerType,
                            cast(TensorType[Any], arg.typ).element_type)
        t = TensorTypeI32.from_type_and_list(element_type, shape)
        return cls.from_input_and_type(arg, t)

    @classmethod
    def from_input_and_type(cls: type[ReshapeOp], arg: SSAValue,
                            t: TensorTypeI32) -> ReshapeOp:
        assert isinstance(arg.typ, TensorType | UnrankedTensorType)
        return cls.create(result_types=[t], operands=[arg])

    def verify_(self):
        result_type = self.res.typ
        assert isinstance(result_type, TensorType)
        result_type = cast(TensorTypeI32, result_type)
        if not len(result_type.shape.data):
            raise VerifyException(
                'Reshape operation result shape should be defined')


@irdl_op_definition
class TransposeOp(Operation, NoSideEffect):
    name: str = 'toy.transpose'
    arguments: Annotated[Operand, AnyTensorTypeI32]
    res: Annotated[OpResult, AnyTensorTypeI32]

    @staticmethod
    def from_input(input: SSAValue):
        input_type = input.typ
        assert isinstance(input_type, TensorType | UnrankedTensorType)
        output_type: TensorType[Any] | UnrankedTensorType[Any]
        if isinstance(input_type, TensorType):
            element_type = cast(IntegerType,
                                cast(TensorType[Any], input_type).element_type)
            output_type = TensorType.from_type_and_list(
                element_type, list(reversed(input_type.shape.data)))
        else:
            output_type = input_type

        return TransposeOp.create(operands=[input], result_types=[output_type])


# Tensor <-> Vector conversion


@irdl_op_definition
class TensorMakeOp(Operation, NoSideEffect):
    name = "toy.tensor.make"

    tensor: Annotated[OpResult, UnrankedTensorTypeI32]
    shape: Annotated[Operand, vd.VectorTypeI32]
    data: Annotated[Operand, vd.VectorTypeI32]

    @classmethod
    def get(cls, shape: Operation | SSAValue, data: Operation | SSAValue,
            result_type: AnyTensorTypeI32) -> TensorMakeOp:
        return cls.build(operands=[shape, data], result_types=[result_type])


@irdl_op_definition
class TensorDataOp(Operation, NoSideEffect):
    name = "toy.tensor.data"

    tensor: Annotated[Operand, AnyTensorTypeI32]
    data: Annotated[OpResult, vd.VectorTypeI32]

    @classmethod
    def get(cls, tensor: Operation | SSAValue) -> TensorDataOp:
        if isinstance(tensor, Operation):
            tensor = tensor.results[0]
        if isinstance(tensor.typ, TensorType):
            data_len = reduce(lambda a, b: a * b, tensor.typ.get_shape(), 1)
        else:
            # Use -1 for unknown length
            data_len = -1
        result_type = vd.VectorTypeI32.from_element_type_and_shape(
            i32, [data_len])
        return cls.build(operands=[tensor], result_types=[result_type])


@irdl_op_definition
class TensorShapeOp(Operation, NoSideEffect):
    name = "toy.tensor.shape"

    tensor: Annotated[Operand, vd.VectorTypeI32]
    data: Annotated[OpResult, AnyTensorTypeI32]

    @classmethod
    def get(cls, tensor: Operation | SSAValue) -> TensorShapeOp:
        if isinstance(tensor, Operation):
            tensor = tensor.results[0]

        # If we know the tensor shape at compile time, we can set the
        # vector length in the type

        if isinstance(tensor.typ, TensorType):
            shape = tensor.typ.get_shape()
        else:
            # Use -1 for unknown length
            shape = [-1]

        result_type = vd.VectorTypeI32.from_element_type_and_shape(i32, shape)

        return cls.build(operands=[tensor], result_types=[result_type])


Toy = Dialect([
    ConstantOp, AddOp, FuncOp, GenericCallOp, PrintOp, MulOp, ReturnOp,
    ReshapeOp, TransposeOp, TensorMakeOp, TensorShapeOp, TensorDataOp
], [])
