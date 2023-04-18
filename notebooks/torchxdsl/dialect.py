from typing import Annotated
from xdsl.dialects.builtin import (
    AnyFloat,
    AnyIntegerAttr,
    ArrayAttr,
    DenseIntOrFPElementsAttr,
    DenseResourceAttr,
    StringAttr,
    AnyFloatAttr,
)
from xdsl.ir import Attribute, Data, Dialect, Operation, OpResult, ParametrizedAttribute
from xdsl.irdl import (
    OpAttr,
    Operand,
    ParameterDef,
    irdl_attr_definition,
    irdl_op_definition,
)
from xdsl.parser import BaseParser, ParserCommons
from xdsl.printer import Printer


@irdl_attr_definition
class BoolAttr(Data[bool]):
    name = "torch.boolattr"

    @staticmethod
    def parse_parameter(parser: BaseParser) -> bool:
        data = parser.expect(parser.try_parse_boolean_literal,
                             "expected boolean")
        if data.text == "true":
            return True
        return False

    def print_parameter(self, printer: Printer) -> None:
        printer.print_string(f"{self.data}")

    @staticmethod
    def from_bool(data: bool):
        return BoolAttr(data)


def parse_torch_type_without_prefix(
        parser: BaseParser) -> ParametrizedAttribute:
    # Get name of type
    type_name = parser.tokenizer.next_token_of_pattern(ParserCommons.bare_id)
    if type_name is None:
        parser.raise_error("Expected a type name")
    type_def = parser.ctx.get_optional_attr(
        "torch." + type_name.text
        if not type_name.text.startswith("torch.") else type_name.text)
    if type_def is None:
        parser.raise_error(f"Unknown type {type_name.text}")

    # Parse parameters
    if issubclass(type_def, ParametrizedAttribute):
        parameters = type_def.parse_parameters(parser)
        return type_def(parameters)

    parser.raise_error(f"Type {type_name.text} is not parametrized")


@irdl_attr_definition
class IntegerType(ParametrizedAttribute):
    name = "torch.int"


@irdl_attr_definition
class FloatType(ParametrizedAttribute):
    name = "torch.float"


@irdl_attr_definition
class BoolType(ParametrizedAttribute):
    name = "torch.bool"


@irdl_attr_definition
class NoneType(ParametrizedAttribute):
    name = "torch.none"


@irdl_attr_definition
class VTensorType(ParametrizedAttribute):
    name = "torch.vtensor"
    dimensions: ParameterDef[ArrayAttr[AnyIntegerAttr]]
    type: ParameterDef[AnyFloat]


@irdl_attr_definition
class ListType(ParametrizedAttribute):
    name = "torch.list"
    type: ParameterDef[IntegerType | FloatType | BoolType]

    @staticmethod
    def parse_parameters(parser: BaseParser) -> list[Attribute]:
        parser.parse_char("<")
        parsed_type = parse_torch_type_without_prefix(parser)
        parser.parse_char(">")
        return [parsed_type]

    def print_paramters(self, printer: Printer) -> None:
        printer.print_string("<")
        printer.print_attribute(self.type)
        printer.print_string(">")


@irdl_op_definition
class ConstantIntOp(Operation):
    name = "torch.constant.int"
    value: OpAttr[AnyIntegerAttr]
    res: Annotated[OpResult, IntegerType]


@irdl_op_definition
class ConstantFloatOp(Operation):
    name = "torch.constant.float"
    value: OpAttr[AnyFloatAttr]
    res: Annotated[OpResult, FloatType]


@irdl_op_definition
class ConstantBoolOp(Operation):
    name = "torch.constant.bool"
    value: OpAttr[AnyIntegerAttr]
    res: Annotated[OpResult, BoolType]


@irdl_op_definition
class ConstantNoneOp(Operation):
    name = "torch.constant.none"
    res: Annotated[OpResult, NoneType]


@irdl_op_definition
class VTensorLitteralOp(Operation):
    name = "torch.vtensor.literal"
    value: OpAttr[DenseIntOrFPElementsAttr | DenseResourceAttr]
    res: Annotated[OpResult, VTensorType]


@irdl_op_definition
class ListConstructOp(Operation):
    name = "torch.prim.ListConstruct"
    lhs: Annotated[Operand, IntegerType]
    rhs: Annotated[Operand, IntegerType]
    res: Annotated[OpResult, ListType]


@irdl_op_definition
class ConvolutionOp(Operation):
    name = "torch.aten.convolution"
    input: Annotated[Operand, VTensorType]
    weight: Annotated[Operand, VTensorType]
    bias: Annotated[Operand, VTensorType]
    padding: Annotated[Operand, ListType]
    stride: Annotated[Operand, ListType]
    dilation: Annotated[Operand, ListType]
    transposed: Annotated[Operand, BoolType]
    output_padding: Annotated[Operand, ListType]
    groups: Annotated[Operand, IntegerType]
    res: Annotated[OpResult, VTensorType]


@irdl_op_definition
class ReluOP(Operation):
    name = "torch.aten.relu"
    arg: Annotated[Operand, VTensorType]
    res: Annotated[OpResult, VTensorType]


@irdl_op_definition
class MaxPool2DOp(Operation):
    name = "torch.aten.max_pool2d"
    input: Annotated[Operand, VTensorType]
    kernel_size: Annotated[Operand, ListType]
    stride: Annotated[Operand, ListType]
    padding: Annotated[Operand, ListType]
    dilation: Annotated[Operand, ListType]
    ceil_mode: Annotated[Operand, BoolType]
    res: Annotated[OpResult, VTensorType]


@irdl_op_definition
class AvgPool2DOp(Operation):
    name = "torch.aten.avg_pool2d"
    input: Annotated[Operand, VTensorType]
    kernel_size: Annotated[Operand, ListType]
    stride: Annotated[Operand, ListType]
    padding: Annotated[Operand, ListType]
    ceil_mode: Annotated[Operand, BoolType]
    count_include_path: Annotated[Operand, BoolType]
    divisor_override: Annotated[Operand, NoneType]
    res: Annotated[OpResult, VTensorType]


@irdl_op_definition
class AssertOp(Operation):
    name = "torch.runtime.assert"
    cond: Annotated[Operand, BoolType]
    message: OpAttr[StringAttr]


@irdl_op_definition
class ViewOp(Operation):
    name = "torch.aten.view"
    tensor: Annotated[Operand, VTensorType]
    size: Annotated[Operand, ListType]
    res: Annotated[OpResult, VTensorType]


@irdl_op_definition
class TransposeOp(Operation):
    name = "torch.aten.transpose.int"
    tensor: Annotated[Operand, VTensorType]
    dim1: Annotated[Operand, IntegerType]
    dim2: Annotated[Operand, IntegerType]
    res: Annotated[OpResult, VTensorType]


@irdl_op_definition
class MMOp(Operation):
    name = "torch.aten.mm"
    lhs: Annotated[Operand, VTensorType]
    rhs: Annotated[Operand, VTensorType]
    res: Annotated[OpResult, VTensorType]


@irdl_op_definition
class AddTensorOp(Operation):
    name = "torch.aten.add.Tensor"
    lhs: Annotated[Operand, VTensorType]
    rhs: Annotated[Operand, VTensorType]
    alpha: Annotated[Operand, FloatType]
    res: Annotated[OpResult, VTensorType]


Torch = Dialect(
    [
        ConstantIntOp,
        ConstantFloatOp,
        ConstantBoolOp,
        ConstantNoneOp,
        VTensorLitteralOp,
        ListConstructOp,
        ConvolutionOp,
        ReluOP,
        MaxPool2DOp,
        AssertOp,
        AvgPool2DOp,
        ViewOp,
        TransposeOp,
        MMOp,
        AddTensorOp,
    ],
    [
        IntegerType,
        FloatType,
        BoolType,
        NoneType,
        VTensorType,
        ListType,
        BoolAttr,
    ],
)
