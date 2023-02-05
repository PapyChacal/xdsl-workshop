from pathlib import Path

from xdsl.ir import MLContext
from xdsl.dialects.builtin import ModuleOp
from xdsl.printer import Printer
from xdsl.pattern_rewriter import PatternRewriteWalker

from riscv.emulator_iop import run_riscv

from toy.dialect import Toy
from toy.mlir_gen import MLIRGen
from toy.parser import Parser

from .accelerator import ToyAccelerator

from .lowering import (AddDataSection, LowerFuncOp, LowerReturnOp,
                       LowerConstantOp, LowerPrintOp, LowerReshapeOp)


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


def lower_toy(module: ModuleOp) -> ModuleOp:
    copy = module.clone()

    PatternRewriteWalker(AddDataSection()).rewrite_module(copy)
    PatternRewriteWalker(LowerFuncOp()).rewrite_module(copy)
    PatternRewriteWalker(LowerReturnOp()).rewrite_module(copy)
    PatternRewriteWalker(LowerConstantOp()).rewrite_module(copy)
    PatternRewriteWalker(LowerPrintOp()).rewrite_module(copy)
    PatternRewriteWalker(LowerReshapeOp()).rewrite_module(copy)

    return copy


def emulate_riscv(program: str):
    run_riscv(program, extensions=[ToyAccelerator], unlimited_regs=True)
