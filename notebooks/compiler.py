from pathlib import Path

from xdsl.ir import MLContext, Operation
from xdsl.dialects.builtin import ModuleOp

from xdsl.printer import Printer
from xdsl.pattern_rewriter import PatternRewriteWalker

from riscv.emulator_iop import run_riscv, print_riscv_ssa

from toy.dialect import Toy
from toy.mlir_gen import MLIRGen
from toy.parser import Parser

from toy.rewrites import (SimplifyRedundantTranspose, RemoveUnusedOperations,
                          ReshapeReshapeOptPattern,
                          FoldConstantReshapeOptPattern)

from toy.lowering import (LowerTensorConstantOp, LowerReshapeOp,
                          LowerTensorAddOp)

from riscv_buffer_ir.accelerator import ToyAccelerator

from toy_to_riscv import (AddSections, LowerFuncOp, LowerReturnOp,
                          LowerPrintOp, LowerVectorConstantOp,
                          LowerTensorMakeOp, LowerTensorShapeOp,
                          LowerTensorDataOp, LowerVectorAddOp, LowerAllocOp)

from vector_ir.rewrites import (SimplifyRedundantShapeAccess,
                                SimplifyRedundantDataAccess)


def parse_toy(program: str, ctx: MLContext | None = None) -> ModuleOp:
    if ctx is None:
        ctx = MLContext()
        ctx.register_dialect(Toy)
    mlir_gen = MLIRGen(ctx)
    module_ast = Parser(Path('in_memory'), program).parseModule()
    module_op = mlir_gen.mlir_gen_module(module_ast)
    return module_op


def print_op(op: Operation):
    Printer(target=Printer.Target.MLIR).print(op)


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

    # PatternRewriteWalker(LowerAllocOp()).rewrite_module(copy)

    return copy


def compile(program: str) -> str:
    toy_0 = parse_toy(program)
    toy_1 = optimise_toy(toy_0)
    vir_0 = lower_from_toy(toy_1)
    vir_1 = optimise_vir(vir_0)
    risc_0 = lower_to_riscv(vir_1)
    code = print_riscv_ssa(risc_0)
    return code


def emulate_riscv(program: str):
    run_riscv(program,
              extensions=[ToyAccelerator],
              unlimited_regs=True,
              verbosity=1)
