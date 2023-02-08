from xdsl.ir import OpResult
from xdsl.pattern_rewriter import (op_type_rewrite_pattern, RewritePattern,
                                   PatternRewriter)

import toy.dialect as td


class SimplifyRedundantShapeAccess(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: td.TensorShapeOp,
                          rewriter: PatternRewriter):
        """
        Fold tensor(t_shape, t_data).data -> t_data
        """
        # Look at the input of the current transpose.
        tensor_data_input = op.tensor
        if not isinstance(tensor_data_input, OpResult):
            # Input was not produced by an operation, could be a function argument
            return

        tensor_make_op = tensor_data_input.op
        if not isinstance(tensor_make_op, td.TensorMakeOp):
            # Input defined by a constant passed in? If not, no match.
            return

        rewriter.replace_matched_op([], [tensor_make_op.shape])


class SimplifyRedundantDataAccess(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: td.TensorDataOp,
                          rewriter: PatternRewriter):
        """
        Fold tensor(t_shape, t_data).data -> t_data
        """
        # Look at the input of the current transpose.
        tensor_data_input = op.tensor
        if not isinstance(tensor_data_input, OpResult):
            # Input was not produced by an operation, could be a function argument
            return

        tensor_make_op = tensor_data_input.op
        if not isinstance(tensor_make_op, td.TensorMakeOp):
            # Input defined by a constant passed in? If not, no match.
            return
        rewriter.replace_matched_op([], [tensor_make_op.data])
