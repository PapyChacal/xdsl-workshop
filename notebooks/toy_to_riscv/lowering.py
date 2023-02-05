from collections import Counter

from xdsl.ir import Operation
from xdsl.dialects.builtin import ModuleOp, UnrankedTensorType
from xdsl.pattern_rewriter import (op_type_rewrite_pattern, RewritePattern,
                                   PatternRewriter)

import toy.dialect as td
import riscv.riscv_ssa as rd
# import toy_to_riscv.dialect as trd


class AddDataSection(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ModuleOp, rewriter: PatternRewriter):

        data_section = rd.DataSectionOp.from_ops([])

        # insert a heap pointer at the start of every function
        rewriter.insert_op_before(data_section, op.regions[0].blocks[0].ops[0])


class LowerFuncOp(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: td.FuncOp, rewriter: PatternRewriter):
        name = op.sym_name.data

        # TODO: add support for user defined functions
        assert name == 'main', 'Only support lowering main function for now'

        region = op.regions[0]

        # insert a heap pointer at the start of every function
        rewriter.insert_op_before(rd.LIOp.get('heap'), region.blocks[0].ops[0])

        # create riscv func op with same ops
        riscv_op = rd.FuncOp.from_region(
            name, rewriter.move_region_contents_to_new_regions(region))

        rewriter.replace_matched_op(riscv_op)


class LowerReturnOp(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: td.ReturnOp, rewriter: PatternRewriter):
        # TODO: add support for optional argument
        assert op.input is None, 'Only support return with no arguments for now'

        rewriter.replace_matched_op(rd.ReturnOp.get())


class LowerConstantOp(RewritePattern):

    _data_section: rd.DataSectionOp | None = None
    _counter: Counter[str] = Counter()

    def data_section(self, op: Operation) -> rd.DataSectionOp:
        '''
        Relies on the data secition being inserted earlier by AddDataSection
        '''
        if self._data_section is None:
            module_op = op.get_toplevel_object()
            assert isinstance(
                module_op, ModuleOp
            ), f'The top level object of {str(op)} must be a ModuleOp'
            first_op = module_op.regions[0].blocks[0].ops[0]
            assert isinstance(first_op, rd.DataSectionOp)
            self._data_section = first_op

        return self._data_section

    def label(self, *args: str) -> str:
        key = '.'.join(args)
        count = self._counter[key]
        self._counter[key] += 1
        return f'{key}.{count}'

    def func_name_of_op(self, op: Operation) -> str:
        region = op.parent_region()
        assert region is not None
        func_op = region.parent_op()
        assert isinstance(func_op, rd.FuncOp)
        return func_op.func_name.data

    def encoded_data(self, shape: list[int], data: list[int]) -> str:
        rank = len(shape)
        count = len(data)
        total = rank + count + 2

        encoded_ints = [total, rank, *shape, count, *data]
        encoded_data = ', '.join(hex(el) for el in encoded_ints)

        return encoded_data

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: td.ConstantOp, rewriter: PatternRewriter):
        value_type = op.value.type

        assert not isinstance(
            value_type,
            UnrankedTensorType), 'Toy constants always have rank information'

        shape: list[int] = value_type.get_shape()
        data: list[int] = [int(el.value.data) for el in op.value.data.data]

        label = self.label(self.func_name_of_op(op))

        data_section = self.data_section(op)

        data_section.regions[0].blocks[0].add_ops([
            rd.LabelOp.get(label),
            rd.DirectiveOp.get(".word", self.encoded_data(shape, data))
        ])

        rewriter.replace_matched_op(rd.LIOp.get(label))
