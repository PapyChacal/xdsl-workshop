from xdsl.ir import OpResult
from xdsl.pattern_rewriter import (op_type_rewrite_pattern, RewritePattern,
                                   PatternRewriter)

import toy_to_riscv.dialect as trd


class SimplifyRedundantShapeAccess(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: trd.TensorShapeOp,
                          rewriter: PatternRewriter):
        """
        Fold tensor(t_shape, t_data).data -> t_data
        """
        # Look at the input of the current transpose.
        tensor_data_input = op.rs1
        if not isinstance(tensor_data_input, OpResult):
            # Input was not produced by an operation, could be a function argument
            return

        tensor_make_op = tensor_data_input.op
        if not isinstance(tensor_make_op, trd.TensorMakeOp):
            # Input defined by a constant passed in? If not, no match.
            return

        rewriter.replace_op(op, [], [tensor_make_op.rs1])


class SimplifyRedundantDataAccess(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: trd.TensorDataOp,
                          rewriter: PatternRewriter):
        """
        Fold tensor(t_shape, t_data).data -> t_data
        """
        # Look at the input of the current transpose.
        tensor_data_input = op.rs1
        if not isinstance(tensor_data_input, OpResult):
            # Input was not produced by an operation, could be a function argument
            return

        tensor_make_op = tensor_data_input.op
        if not isinstance(tensor_make_op, trd.TensorMakeOp):
            # Input defined by a constant passed in? If not, no match.
            return

        rewriter.replace_op(op, [], [tensor_make_op.rs2])
