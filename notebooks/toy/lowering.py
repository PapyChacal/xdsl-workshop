from xdsl.dialects.builtin import UnrankedTensorType, TensorType
from xdsl.pattern_rewriter import (op_type_rewrite_pattern, RewritePattern,
                                   PatternRewriter)

import toy.dialect as td
import riscv_buffer_ir.dialect as rbd
import vector_ir.dialect as tvd


class LowerTensorConstantOp(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: td.ConstantOp, rewriter: PatternRewriter):
        value_type = op.value.type

        assert not isinstance(
            value_type,
            UnrankedTensorType), 'Toy constants always have rank information'

        shape: list[int] = value_type.get_shape()
        data: list[int] = [int(el.value.data) for el in op.value.data.data]

        shape_vector = tvd.VectorConstantOp.get(shape, 'tensor_shape')
        data_vector = tvd.VectorConstantOp.get(data, 'tensor_data')
        tensor = tvd.TensorMakeOp.get(shape_vector, data_vector)

        rewriter.replace_matched_op([shape_vector, data_vector, tensor])


class LowerPrintOp(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: td.PrintOp, rewriter: PatternRewriter):
        rewriter.replace_matched_op(rbd.TensorPrintOp.get(op.input))


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
            new_shape := tvd.VectorConstantOp.get(shape, 'tensor_new_shape'),
            old_data := tvd.TensorDataOp.get(op.arg),
            tvd.TensorMakeOp.get(new_shape, old_data)
        ])


class LowerTensorAddOp(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: td.AddOp, rewriter: PatternRewriter):
        shape = tvd.TensorShapeOp.get(op.lhs)
        lhs = tvd.TensorDataOp.get(op.lhs)
        rhs = tvd.TensorDataOp.get(op.rhs)
        sum = tvd.VectorAddOp.get(lhs, rhs)
        result = tvd.TensorMakeOp.get(shape, sum)

        rewriter.replace_matched_op([shape, lhs, rhs, sum, result])
