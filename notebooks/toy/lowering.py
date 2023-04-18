from typing import cast
from xdsl.dialects.builtin import UnrankedTensorType, TensorType
from xdsl.pattern_rewriter import (op_type_rewrite_pattern, RewritePattern,
                                   PatternRewriter)

import toy.dialect as td
import riscv_buffer_ir.dialect as rbd
import vector_ir.dialect as tvd


class LowerTensorConstantOp(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: td.ConstantOp, rewriter: PatternRewriter):
        typ = op.value.type

        assert isinstance(
            typ, TensorType), 'Toy constants always have rank information'
        typ = cast(td.AnyTensorTypeI32, typ)

        shape: list[int] = op.get_shape()
        data: list[int] = op.get_data()

        shape_vector = tvd.VectorConstantOp.get(shape, 'tensor_shape')
        data_vector = tvd.VectorConstantOp.get(data, 'tensor_data')
        tensor = td.TensorMakeOp.get(shape_vector, data_vector, typ)

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
        typ = cast(td.TensorTypeI32, typ)
        shape = typ.get_shape()

        rewriter.replace_matched_op([
            new_shape := tvd.VectorConstantOp.get(shape, 'tensor_new_shape'),
            old_data := td.TensorDataOp.get(op.arg),
            td.TensorMakeOp.get(new_shape, old_data, typ)
        ])


class LowerTensorAddOp(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: td.AddOp, rewriter: PatternRewriter):
        typ = op.res.typ
        assert isinstance(typ, TensorType | UnrankedTensorType)
        typ = cast(td.AnyTensorTypeI32, typ)

        shape = td.TensorShapeOp.get(op.lhs)
        lhs = td.TensorDataOp.get(op.lhs)
        rhs = td.TensorDataOp.get(op.rhs)
        sum = tvd.VectorAddOp.get(lhs, rhs)
        result = td.TensorMakeOp.get(shape, sum, typ)

        rewriter.replace_matched_op([shape, lhs, rhs, sum, result])
