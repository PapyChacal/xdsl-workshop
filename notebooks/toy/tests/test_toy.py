from pathlib import Path
from io import StringIO

from xdsl.dialects.builtin import i32, ModuleOp, UnrankedTensorType
from xdsl.ir import BlockArgument, MLContext, Operation
from xdsl.pattern_rewriter import (GreedyRewritePatternApplier,
                                   PatternRewriteWalker)
from xdsl.printer import Printer

from ..parser import Parser
from ..toy_ast import (ModuleAST, FunctionAST, PrototypeAST, VariableExprAST,
                       ReturnExprAST, BinaryExprAST, CallExprAST,
                       VarDeclExprAST, VarType, LiteralExprAST, NumberExprAST)
from ..location import Location
from ..mlir_gen import MLIRGen
from ..dialect import (ConstantOp, FuncOp, GenericCallOp, MulOp, PrintOp,
                       ReturnOp, ReshapeOp, TransposeOp)
from ..rewrites import ReshapeReshapeOptPattern, SimplifyRedundantTranspose, RemoveUnusedOperations, FoldConstantReshapeOptPattern


def test_parse_ast():
    ast_toy = Path('notebooks/examples/ast.toy')

    with open(ast_toy, 'r') as f:
        parser = Parser(ast_toy, f.read())

    parsed_module_ast = parser.parseModule()

    def loc(line: int, col: int) -> Location:
        return Location(ast_toy, line, col)

    module_ast = ModuleAST((
        FunctionAST(
            loc(4, 1),
            PrototypeAST(loc(4, 1), 'multiply_transpose', [
                VariableExprAST(loc(4, 24), 'a'),
                VariableExprAST(loc(4, 27), 'b'),
            ]), (ReturnExprAST(
                loc(5, 3),
                BinaryExprAST(
                    loc(5, 25), '*',
                    CallExprAST(loc(5, 10), 'transpose',
                                [VariableExprAST(loc(5, 20), 'a')]),
                    CallExprAST(loc(5, 25), 'transpose',
                                [VariableExprAST(loc(5, 35), 'b')]))), )),
        FunctionAST(loc(8, 1), PrototypeAST(loc(8, 1), 'main', []), (
            VarDeclExprAST(
                loc(11, 3), 'a', VarType([]),
                LiteralExprAST(loc(11, 11), [
                    LiteralExprAST(loc(11, 12), [
                        NumberExprAST(loc(11, 13), 1),
                        NumberExprAST(loc(11, 16), 2),
                        NumberExprAST(loc(11, 19), 3)
                    ], [3]),
                    LiteralExprAST(loc(11, 23), [
                        NumberExprAST(loc(11, 24), 4),
                        NumberExprAST(loc(11, 27), 5),
                        NumberExprAST(loc(11, 30), 6),
                    ], [3])
                ], [2, 3])),
            VarDeclExprAST(
                loc(15, 3), 'b', VarType([2, 3]),
                LiteralExprAST(loc(15, 17), [
                    NumberExprAST(loc(15, 18), 1),
                    NumberExprAST(loc(15, 21), 2),
                    NumberExprAST(loc(15, 24), 3),
                    NumberExprAST(loc(15, 27), 4),
                    NumberExprAST(loc(15, 30), 5),
                    NumberExprAST(loc(15, 33), 6),
                ], [6])),
            VarDeclExprAST(
                loc(19, 3), 'c', VarType([]),
                CallExprAST(loc(19, 11), 'multiply_transpose', [
                    VariableExprAST(loc(19, 30), 'a'),
                    VariableExprAST(loc(19, 33), 'b'),
                ])),
            VarDeclExprAST(
                loc(22, 3), 'd', VarType([]),
                CallExprAST(loc(22, 11), 'multiply_transpose', [
                    VariableExprAST(loc(22, 30), 'b'),
                    VariableExprAST(loc(22, 33), 'a'),
                ])),
            VarDeclExprAST(
                loc(25, 3), 'e', VarType([]),
                CallExprAST(loc(25, 11), 'multiply_transpose', [
                    VariableExprAST(loc(25, 30), 'b'),
                    VariableExprAST(loc(25, 33), 'c'),
                ])),
            VarDeclExprAST(
                loc(28, 3), 'f', VarType([]),
                CallExprAST(loc(28, 11), 'multiply_transpose', [
                    CallExprAST(loc(28, 30), 'transpose',
                                [VariableExprAST(loc(28, 40), 'a')]),
                    VariableExprAST(loc(28, 44), 'c'),
                ])),
        )),
    ))

    assert parsed_module_ast == module_ast


def test_convert_ast():
    ast_toy = Path('notebooks/examples/ast.toy')

    with open(ast_toy, 'r') as f:
        parser = Parser(ast_toy, f.read())

    module_ast = parser.parseModule()

    ctx = MLContext()
    mlir_gen = MLIRGen(ctx)

    generated_module_op = mlir_gen.mlir_gen_module(module_ast)

    unrankedi32TensorType = UnrankedTensorType.from_type(i32)

    def func_body(*args: BlockArgument) -> list[Operation]:
        arg0, arg1 = args
        f0 = TransposeOp.from_input(arg0)
        f1 = TransposeOp.from_input(arg1)
        f2 = MulOp.from_summands(f0.results[0], f1.results[0])
        f3 = ReturnOp.from_input(f2.results[0])
        return [f0, f1, f2, f3]

    def main_body(*args: BlockArgument) -> list[Operation]:
        m0 = ConstantOp.from_list([1, 2, 3, 4, 5, 6], [2, 3])
        [a] = m0.results
        m1 = ConstantOp.from_list([1, 2, 3, 4, 5, 6], [6])
        m2 = ReshapeOp.from_input(m1.results[0], [2, 3])
        [b] = m2.results
        m3 = GenericCallOp.get('multiply_transpose', [a, b],
                               [unrankedi32TensorType])
        [c] = m3.results
        m4 = GenericCallOp.get('multiply_transpose', [b, a],
                               [unrankedi32TensorType])
        m5 = GenericCallOp.get('multiply_transpose', [b, c],
                               [unrankedi32TensorType])
        m6 = TransposeOp.from_input(a)
        [a_transposed] = m6.results
        m7 = GenericCallOp.get('multiply_transpose', [a_transposed, c],
                               [unrankedi32TensorType])
        m8 = ReturnOp.from_input()
        return [m0, m1, m2, m3, m4, m5, m6, m7, m8]

    multiply_transpose = FuncOp.from_callable(
        'multiply_transpose', [unrankedi32TensorType, unrankedi32TensorType],
        [unrankedi32TensorType],
        func_body,
        private=True)
    main = FuncOp.from_callable('main', [], [], main_body, private=False)

    module_op = ModuleOp.from_region_or_ops([multiply_transpose, main])

    assert module_op.is_structurally_equivalent(generated_module_op)


def test_rewrite_transposes():
    example = """
    def transpose_transpose(x) {
        return transpose(transpose(x));
    }
    """

    ctx = MLContext()
    mlir_gen = MLIRGen(ctx)

    module_ast = Parser(Path('in_memory'), example).parseModule()

    generated_module_op = mlir_gen.mlir_gen_module(module_ast)

    unrankedi32TensorType = UnrankedTensorType.from_type(i32)

    def func_body_0(*args: BlockArgument) -> list[Operation]:
        arg0, = args
        f0 = TransposeOp.from_input(arg0)
        f1 = TransposeOp.from_input(f0.results[0])
        f2 = ReturnOp.from_input(f1.results[0])
        return [f0, f1, f2]

    def func_body_1(*args: BlockArgument) -> list[Operation]:
        arg0, = args
        f0 = TransposeOp.from_input(arg0)
        f1 = ReturnOp.from_input(arg0)
        return [f0, f1]

    def func_body_2(*args: BlockArgument) -> list[Operation]:
        arg0, = args
        f0 = ReturnOp.from_input(arg0)
        return [f0]

    assert generated_module_op.is_structurally_equivalent(
        ModuleOp.from_region_or_ops([
            FuncOp.from_callable('transpose_transpose',
                                 [unrankedi32TensorType],
                                 [unrankedi32TensorType],
                                 func_body_0,
                                 private=True)
        ]))

    PatternRewriteWalker(
        GreedyRewritePatternApplier([SimplifyRedundantTranspose()
                                     ])).rewrite_module(generated_module_op)

    assert generated_module_op.is_structurally_equivalent(
        ModuleOp.from_region_or_ops([
            FuncOp.from_callable('transpose_transpose',
                                 [unrankedi32TensorType],
                                 [unrankedi32TensorType],
                                 func_body_1,
                                 private=True)
        ]))

    PatternRewriteWalker(
        GreedyRewritePatternApplier([RemoveUnusedOperations()
                                     ])).rewrite_module(generated_module_op)

    assert generated_module_op.is_structurally_equivalent(
        ModuleOp.from_region_or_ops([
            FuncOp.from_callable('transpose_transpose',
                                 [unrankedi32TensorType],
                                 [unrankedi32TensorType],
                                 func_body_2,
                                 private=True)
        ]))


def test_constant_folding():
    example = """
    def main() {
        var a<2,1> = [1, 2];
        var b<2,1> = a;
        var c<2,1> = b;
        print(c);
    }
    """

    ctx = MLContext()
    mlir_gen = MLIRGen(ctx)

    module_ast = Parser(Path('in_memory'), example).parseModule()

    generated_module_op = mlir_gen.mlir_gen_module(module_ast)

    def main_0(*args: BlockArgument) -> list[Operation]:
        _ = args
        m0 = ConstantOp.from_list([1, 2], [2])
        m1 = ReshapeOp.from_input(m0.res, [2, 1])
        m2 = ReshapeOp.from_input(m1.res, [2, 1])
        m3 = ReshapeOp.from_input(m2.res, [2, 1])
        m4 = PrintOp.from_input(m3.res)
        m5 = ReturnOp.from_input()
        return [m0, m1, m2, m3, m4, m5]

    def main_1(*args: BlockArgument) -> list[Operation]:
        _ = args
        m0 = ConstantOp.from_list([1, 2], [2])
        m1 = ReshapeOp.from_input(m0.results[0], [2, 1])
        m2 = ReshapeOp.from_input(m0.results[0], [2, 1])
        m3 = ReshapeOp.from_input(m0.results[0], [2, 1])
        m4 = PrintOp.from_input(m3.res)
        m5 = ReturnOp.from_input()
        return [m0, m1, m2, m3, m4, m5]

    def main_2(*args: BlockArgument) -> list[Operation]:
        _ = args
        m0 = ConstantOp.from_list([1, 2], [2])
        m1 = ReshapeOp.from_input(m0.res, [2, 1])
        m2 = PrintOp.from_input(m1.res)
        m3 = ReturnOp.from_input()
        return [m0, m1, m2, m3]

    def main_3(*args: BlockArgument) -> list[Operation]:
        _ = args
        m0 = ConstantOp.from_list([1, 2], [2])
        m1 = ConstantOp.from_list([1, 2], [2, 1])
        m2 = PrintOp.from_input(m1.res)
        m3 = ReturnOp.from_input()
        return [m0, m1, m2, m3]

    def main_4(*args: BlockArgument) -> list[Operation]:
        _ = args
        m0 = ConstantOp.from_list([1, 2], [2, 1])
        m1 = PrintOp.from_input(m0.res)
        m2 = ReturnOp.from_input()
        return [m0, m1, m2]

    assert generated_module_op.is_structurally_equivalent(
        ModuleOp.from_region_or_ops(
            [FuncOp.from_callable('main', [], [], main_0, private=False)]))

    PatternRewriteWalker(
        GreedyRewritePatternApplier([ReshapeReshapeOptPattern()
                                     ])).rewrite_module(generated_module_op)

    assert generated_module_op.is_structurally_equivalent(
        ModuleOp.from_region_or_ops(
            [FuncOp.from_callable('main', [], [], main_1, private=False)]))

    PatternRewriteWalker(
        GreedyRewritePatternApplier([RemoveUnusedOperations()
                                     ])).rewrite_module(generated_module_op)

    assert generated_module_op.is_structurally_equivalent(
        ModuleOp.from_region_or_ops(
            [FuncOp.from_callable('main', [], [], main_2, private=False)]))

    PatternRewriteWalker(
        GreedyRewritePatternApplier([FoldConstantReshapeOptPattern()
                                     ])).rewrite_module(generated_module_op)

    assert generated_module_op.is_structurally_equivalent(
        ModuleOp.from_region_or_ops(
            [FuncOp.from_callable('main', [], [], main_3, private=False)]))

    PatternRewriteWalker(
        GreedyRewritePatternApplier([RemoveUnusedOperations()
                                     ])).rewrite_module(generated_module_op)

    assert generated_module_op.is_structurally_equivalent(
        ModuleOp.from_region_or_ops(
            [FuncOp.from_callable('main', [], [], main_4, private=False)]))
