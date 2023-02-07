from xdsl.dialects.builtin import UnrankedTensorType, TensorType
from xdsl.pattern_rewriter import (op_type_rewrite_pattern, RewritePattern,
                                   PatternRewriter)

import toy.dialect as td
import toy_to_riscv.dialect as trd


class LowerTensorConstantOp(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: td.ConstantOp, rewriter: PatternRewriter):
        value_type = op.value.type

        assert not isinstance(
            value_type,
            UnrankedTensorType), 'Toy constants always have rank information'

        shape: list[int] = value_type.get_shape()
        data: list[int] = [int(el.value.data) for el in op.value.data.data]

        shape_vector = trd.VectorConstantOp.get(shape, 'tensor_shape')
        data_vector = trd.VectorConstantOp.get(data, 'tensor_data')
        tensor = trd.TensorMakeOp.get(shape_vector, data_vector)

        rewriter.replace_matched_op([shape_vector, data_vector, tensor])


class LowerPrintOp(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: td.PrintOp, rewriter: PatternRewriter):
        rewriter.replace_matched_op(trd.TensorPrintOp.get(op.input))


class LowerReshapeOp(RewritePattern):

    def shape_data(self, shape: list[int]) -> list[int]:
        rank = len(shape)
        encoded_ints = [rank, *shape]
        return encoded_ints

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: td.ReshapeOp, rewriter: PatternRewriter):
        typ = op.res.typ
        assert isinstance(typ, TensorType)
        shape = typ.get_shape()

        rewriter.replace_matched_op([
            new_shape := trd.VectorConstantOp.get(shape, 'tensor_new_shape'),
            old_data := trd.TensorDataOp.get(op.arg),
            trd.TensorMakeOp.get(new_shape, old_data)
        ])


class LowerTensorAddOp(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: td.AddOp, rewriter: PatternRewriter):
        shape = trd.TensorShapeOp.get(op.lhs)
        lhs = trd.TensorDataOp.get(op.lhs)
        rhs = trd.TensorDataOp.get(op.rhs)
        sum = trd.VectorAddOp.get(lhs, rhs)
        result = trd.TensorMakeOp.get(shape, sum)

        rewriter.replace_matched_op([shape, lhs, rhs, sum, result])
