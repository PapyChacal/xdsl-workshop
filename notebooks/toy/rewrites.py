from typing import cast
from xdsl.ir import OpResult, Operation
from xdsl.dialects.builtin import DenseIntOrFPElementsAttr
from xdsl.pattern_rewriter import (op_type_rewrite_pattern, RewritePattern,
                                   PatternRewriter)
from .dialect import ConstantOp, ReshapeOp, TensorTypeI32, TransposeOp, NoSideEffect


class SimplifyRedundantTranspose(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TransposeOp, rewriter: PatternRewriter):
        """
        Fold transpose(transpose(x)) -> x
        """
        # Look at the input of the current transpose.
        transpose_input = op.arguments
        if not isinstance(transpose_input, OpResult):
            # Input was not produced by an operation, could be a function argument
            return

        transpose_input_op = transpose_input.op
        if not isinstance(transpose_input_op, TransposeOp):
            # Input defined by another transpose? If not, no match.
            return

        rewriter.replace_op(op, [], [transpose_input_op.arguments])


# Create our rewriter class:
class RemoveUnusedOperations(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        """
        Removes operations whose result is not used, and that don't have side effects
        """
        # Check that operation is side-effect-free
        if not isinstance(op, NoSideEffect):
            return

        # Look through the input of the current transpose.
        results = op.results
        for result in results:
            if len(result.uses):
                # At least one of the results is used
                return

        rewriter.erase_op(op)


class ReshapeReshapeOptPattern(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ReshapeOp, rewriter: PatternRewriter):
        """
        Reshape(Reshape(x)) = Reshape(x)
        """
        # Look at the input of the current reshape.
        reshape_input = op.arg
        if not isinstance(reshape_input, OpResult):
            # Input was not produced by an operation, could be a function argument
            return

        reshape_input_op = reshape_input.op
        if not isinstance(reshape_input_op, ReshapeOp):
            # Input defined by another transpose? If not, no match.
            return

        t = cast(TensorTypeI32, op.res.typ)
        new_op = ReshapeOp.from_input_and_type(reshape_input_op.arg, t)
        rewriter.replace_matched_op(new_op)


class FoldConstantReshapeOptPattern(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ReshapeOp, rewriter: PatternRewriter):
        """
        Reshaping a constant can be done at compile time
        """
        # Look at the input of the current reshape.
        reshape_input = op.arg
        if not isinstance(reshape_input, OpResult):
            # Input was not produced by an operation, could be a function argument
            return

        reshape_input_op = reshape_input.op
        if not isinstance(reshape_input_op, ConstantOp):
            # Input defined by another transpose? If not, no match.
            return

        new_value = DenseIntOrFPElementsAttr.create_dense_int(
            type=op.res.typ, data=reshape_input_op.value.data.data)
        new_op = ConstantOp.from_value(new_value)
        rewriter.replace_matched_op(new_op)
