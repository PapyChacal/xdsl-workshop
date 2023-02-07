from collections import Counter

from xdsl.ir import Operation, SSAValue, Block
from xdsl.dialects.builtin import ModuleOp, UnrankedTensorType, TensorType
from xdsl.pattern_rewriter import (op_type_rewrite_pattern, RewritePattern,
                                   PatternRewriter)

import toy.dialect as td
import riscv.riscv_ssa as rd
import toy_to_riscv.dialect as trd


class AddSections(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ModuleOp, rewriter: PatternRewriter):

        # bss stands for block starting symbol
        heap_section = rd.SectionOp.from_ops(
            '.bss',
            [
                rd.LabelOp.get("heap"),
                rd.DirectiveOp.get(".space", f'{1024}'),  # 1kb
            ])
        data_section = rd.SectionOp.from_ops('.data', [])
        text_section = rd.SectionOp.from_region(
            '.text',
            rewriter.move_region_contents_to_new_regions(op.regions[0]))

        op.body.add_block(
            Block.from_ops([heap_section, data_section, text_section]))


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


class DataSectionRewritePattern(RewritePattern):

    _data_section: rd.SectionOp | None = None
    _counter: Counter[str] = Counter()

    def data_section(self, op: Operation) -> rd.SectionOp:
        '''
        Relies on the data secition being inserted earlier by AddDataSection
        '''
        if self._data_section is None:
            module_op = op.get_toplevel_object()
            assert isinstance(
                module_op, ModuleOp
            ), f'The top level object of {str(op)} must be a ModuleOp'

            for op in module_op.body.blocks[0].ops:
                if not isinstance(op, rd.SectionOp):
                    continue
                if op.directive.data != '.data':
                    continue
                self._data_section = op

            assert self._data_section is not None

        return self._data_section

    def label(self, func_name: str, kind: str) -> str:
        key = f'{func_name}.{kind}'
        count = self._counter[key]
        self._counter[key] += 1
        return f'{key}.{count}'

    def func_name_of_op(self, op: Operation) -> str:
        region = op.parent_region()
        assert region is not None
        func_op = region.parent_op()
        assert isinstance(func_op, rd.FuncOp)
        return func_op.func_name.data

    def add_data(self, op: Operation, label: str, data: list[int]):
        encoded_data = ', '.join(hex(el) for el in data)
        self.data_section(op).regions[0].blocks[0].add_ops(
            [rd.LabelOp.get(label),
             rd.DirectiveOp.get(".word", encoded_data)])


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


class LowerVectorConstantOp(DataSectionRewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: trd.VectorConstantOp,
                          rewriter: PatternRewriter):
        """
        Vectors are represented in memory as an n+1 array of int32, where the first
        entry is the count of the vector
        """
        data = op.get_data()
        label = self.label(self.func_name_of_op(op), op.label.data)

        self.add_data(op, label, [len(data), *data])
        rewriter.replace_matched_op(rd.LIOp.get(label))


class LowerPrintOp(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: td.PrintOp, rewriter: PatternRewriter):
        rewriter.replace_matched_op(trd.TensorPrintOp.get(op.input))


class LowerReshapeOp(DataSectionRewritePattern):

    def shape_data(self, shape: list[int]) -> list[int]:
        rank = len(shape)
        encoded_ints = [rank, *shape]
        return encoded_ints

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: td.ReshapeOp, rewriter: PatternRewriter):

        label = self.label(self.func_name_of_op(op), 'shape')

        typ = op.res.typ
        assert isinstance(typ, TensorType)
        shape = typ.get_shape()

        self.add_data(op, label, self.shape_data(shape))

        rewriter.replace_matched_op([
            shape := rd.LIOp.get(label),
            data := trd.TensorShapeOp.get(op.arg),
            trd.TensorMakeOp.get(shape, data),
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


class LowerVectorAddOp(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: trd.VectorAddOp,
                          rewriter: PatternRewriter):

        rewriter.replace_matched_op([
            count := rd.LWOp.get(op.rs1, 0, 'Get input count'),
            storage_count := rd.AddIOp.get(count, 4,
                                           'Input storage int32 count'),
            vector := trd.AllocOp.get(storage_count),
            rd.SWOp.get(count, vector, 0, 'Set result count'),
            lhs := rd.AddIOp.get(op.rs1, 4, 'lhs storage'),
            rhs := rd.AddIOp.get(op.rs2, 4, 'lhs storage'),
            dest := rd.AddIOp.get(vector, 4, 'destination storage'),
            trd.BufferAddOp.get(count, lhs, dest),
            trd.BufferAddOp.get(count, rhs, dest),
        ], [vector.rd])


class LowerTensorMakeOp(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: trd.TensorMakeOp,
                          rewriter: PatternRewriter):
        shape = op.rs1
        data = op.rs2

        tensor_storage_len_op = rd.LIOp.get(2, 'Tensor storage')
        tensor_op = trd.AllocOp.get(tensor_storage_len_op)
        tensor_set_shape_op = rd.SWOp.get(shape, tensor_op, 0,
                                          'Set tensor shape')
        tensor_set_data_op = rd.SWOp.get(data, tensor_op, 4, 'Set tensor data')

        rewriter.replace_matched_op([
            tensor_storage_len_op,
            tensor_op,
            tensor_set_shape_op,
            bla := rd.LWOp.get(tensor_op, 0),
            tensor_set_data_op,
        ], [tensor_op.rd])


class LowerTensorShapeOp(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: trd.TensorShapeOp,
                          rewriter: PatternRewriter):
        rewriter.replace_matched_op(rd.LWOp.get(op.rs1, 0, 'Get tensor shape'))


class LowerTensorDataOp(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: trd.TensorDataOp,
                          rewriter: PatternRewriter):
        rewriter.replace_matched_op(rd.LWOp.get(op.rs1, 4, 'Get tensor data'))


class LowerAllocOp(RewritePattern):

    def heap_address(self, op: Operation) -> SSAValue:
        block = op.parent_block()
        assert block is not None
        heap_op = block.ops[0]
        # TODO: check that this is indeed the heap op
        # assert isinstance(heap_op, rd.LIOp) and isinstance(heap_op.immediate, rd.LabelAttr)
        return heap_op.results[0]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: trd.AllocOp, rewriter: PatternRewriter):
        heap_ptr = self.heap_address(op)

        rewriter.replace_matched_op([
            four := rd.LIOp.get(4, '4 bytes per int'),
            count := rd.MULOp.get(op.rs1, four, 'Alloc count bytes'),
            old_heap_count := rd.LWOp.get(heap_ptr, 0, 'Old heap count'),
            new_heap_count := rd.AddOp.get(old_heap_count, count,
                                           'New heap count'),
            rd.SWOp.get(new_heap_count, heap_ptr, 0, 'Update heap'),
            heap_storage_start := rd.AddIOp.get(heap_ptr, 4,
                                                'Heap storage start'),
            result := rd.AddOp.get(heap_storage_start, old_heap_count,
                                   'Allocated memory'),
        ], [result.rd])
