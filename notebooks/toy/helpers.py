from pathlib import Path

from xdsl.ir import MLContext
from xdsl.dialects.builtin import ModuleOp
from xdsl.printer import Printer

from .dialect import Toy
from .mlir_gen import MLIRGen
from .parser import Parser


def parse(program: str, ctx: MLContext | None = None) -> ModuleOp:
    if ctx is None:
        ctx = MLContext()
        ctx.register_dialect(Toy)
    mlir_gen = MLIRGen(ctx)
    module_ast = Parser(Path('in_memory'), program).parseModule()
    module_op = mlir_gen.mlir_gen_module(module_ast)
    return module_op


def print_module(module: ModuleOp):
    Printer(target=Printer.Target.MLIR).print(module)
