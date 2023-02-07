from pathlib import Path

from xdsl.ir import MLContext
from xdsl.dialects.builtin import ModuleOp
from xdsl.printer import Printer
from xdsl.pattern_rewriter import PatternRewriteWalker

from riscv.emulator_iop import run_riscv

from toy.dialect import Toy
from toy.mlir_gen import MLIRGen
from toy.parser import Parser

from toy.rewrites import (SimplifyRedundantTranspose, RemoveUnusedOperations,
                          ReshapeReshapeOptPattern,
                          FoldConstantReshapeOptPattern)

from .accelerator import ToyAccelerator

from .lower_from_toy import (AddSections, LowerFuncOp, LowerReturnOp,
                             LowerTensorConstantOp, LowerPrintOp,
                             LowerReshapeOp, LowerTensorAddOp,
                             LowerVectorConstantOp, LowerTensorMakeOp,
                             LowerAllocOp, LowerTensorShapeOp,
                             LowerTensorDataOp, LowerVectorAddOp)

from .optimise import (SimplifyRedundantShapeAccess,
                       SimplifyRedundantDataAccess)


def parse_toy(program: str, ctx: MLContext | None = None) -> ModuleOp:
    if ctx is None:
        ctx = MLContext()
        ctx.register_dialect(Toy)
    mlir_gen = MLIRGen(ctx)
    module_ast = Parser(Path('in_memory'), program).parseModule()
    module_op = mlir_gen.mlir_gen_module(module_ast)
    return module_op


def print_module(module: ModuleOp):
    Printer(target=Printer.Target.MLIR).print(module)


def optimise_toy(module: ModuleOp) -> ModuleOp:
    copy = module.clone()

    PatternRewriteWalker(SimplifyRedundantTranspose()).rewrite_module(copy)
    PatternRewriteWalker(RemoveUnusedOperations()).rewrite_module(copy)
    PatternRewriteWalker(ReshapeReshapeOptPattern()).rewrite_module(copy)
    PatternRewriteWalker(FoldConstantReshapeOptPattern()).rewrite_module(copy)
    PatternRewriteWalker(RemoveUnusedOperations()).rewrite_module(copy)

    return copy


def lower_from_toy(module: ModuleOp) -> ModuleOp:
    copy = module.clone()

    PatternRewriteWalker(LowerTensorConstantOp()).rewrite_module(copy)
    PatternRewriteWalker(LowerReshapeOp()).rewrite_module(copy)
    PatternRewriteWalker(LowerTensorAddOp()).rewrite_module(copy)
    PatternRewriteWalker(LowerTensorAddOp()).rewrite_module(copy)

    return copy


def optimise_vir(module: ModuleOp) -> ModuleOp:
    copy = module.clone()

    PatternRewriteWalker(SimplifyRedundantShapeAccess()).rewrite_module(copy)
    PatternRewriteWalker(SimplifyRedundantDataAccess()).rewrite_module(copy)
    PatternRewriteWalker(RemoveUnusedOperations()).rewrite_module(copy)

    return copy


def lower_to_riscv(module: ModuleOp) -> ModuleOp:
    copy = module.clone()

    PatternRewriteWalker(AddSections()).rewrite_module(copy)
    PatternRewriteWalker(LowerFuncOp()).rewrite_module(copy)
    PatternRewriteWalker(LowerReturnOp()).rewrite_module(copy)
    PatternRewriteWalker(LowerPrintOp()).rewrite_module(copy)

    PatternRewriteWalker(LowerVectorConstantOp()).rewrite_module(copy)
    PatternRewriteWalker(LowerTensorMakeOp()).rewrite_module(copy)
    PatternRewriteWalker(LowerTensorShapeOp()).rewrite_module(copy)
    PatternRewriteWalker(LowerTensorDataOp()).rewrite_module(copy)
    PatternRewriteWalker(LowerVectorAddOp()).rewrite_module(copy)

    return copy


def emulate_riscv(program: str):
    run_riscv(program, extensions=[ToyAccelerator], unlimited_regs=True)
