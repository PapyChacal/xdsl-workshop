from __future__ import annotations

# pyright: reportMissingTypeStubs=false

from riscemu import RunConfig, UserModeCPU, RV32I, RV32M, AssemblyFileLoader, MMU
from riscemu.instructions import InstructionSet, Instruction

from io import StringIO
from typing import Any, List, Type, cast, Generator

from xdsl.ir import Operation
from xdsl.dialects.builtin import ModuleOp
from .riscv_ssa import (FuncOp, SSAValue, LabelAttr, IntegerAttr, DirectiveOp,
                        LabelOp, RiscvNoParamsOperation, ECALLOp, SectionOp,
                        ReturnOp, Riscv2Rs1ImmOperation,
                        Riscv1Rd1Rs1ImmOperation, Riscv1Rd2RsOperation)

SCALL_EXIT = 93


class RV_Debug(InstructionSet):

    # this instruction will dissappear into our emualtor soon-ish
    def instruction_print(self, ins: Instruction):
        reg = ins.get_reg(0)
        value = self.regs.get(reg)  # pyright: ignore[reportUnknownMemberType]
        print(f"register {reg} contains value {value}")


class _SSAVALNamer:

    ssa_val_names: dict[SSAValue, int] = {}
    idx = 0

    def __init__(self) -> None:
        self.ssa_val_names = dict()
        self.idx = 0

    def get_ssa_name(self, reg: SSAValue):
        if reg not in self.ssa_val_names:
            id = self.idx
            self.idx += 1
            self.ssa_val_names[reg] = id
        return str('%' + str(self.ssa_val_names[reg]))


def riscv_ops(module: ModuleOp) -> Generator[Operation, None, None]:
    for op in module.body.blocks[0].ops:
        assert isinstance(op, SectionOp)
        yield DirectiveOp.get(op.directive, '')
        for op in op.data.blocks[0].ops:
            if isinstance(op, FuncOp):
                yield LabelOp.get(op.func_name.data)
                yield from op.func_body.blocks[0].ops
                if op.func_name.data == 'main':
                    # Exit syscall at the end of 'main'
                    yield ECALLOp.get(SCALL_EXIT)
            else:
                yield op


def print_riscv_ssa(module: ModuleOp, memory: int = 1024) -> str:
    ops = list(riscv_ops(module))

    out = ""
    reg = _SSAVALNamer()

    def get_all_regs(op: Operation):
        for name in ('rd', 'rs', 'rt', 'rs1', 'rs2', 'rs3', 'offset',
                     'immediate'):
            if hasattr(op, name):
                val = getattr(op, name)
                if isinstance(val, SSAValue):
                    yield reg.get_ssa_name(val)
                elif isinstance(val, LabelAttr):
                    yield val.data
                elif isinstance(val, IntegerAttr):
                    yield str(val.value.data)
                else:
                    yield val.data

    for op in ops:
        name = '.'.join(op.name.split(".")[1:])
        if isinstance(op, DirectiveOp):
            out += "{} {}".format(op.directive.data, op.value.data)
        elif isinstance(op, LabelOp):
            out += "{}:".format(op.label.data)
        elif isinstance(op, RiscvNoParamsOperation):
            out += "{}".format(name)
        elif isinstance(op, ECALLOp):
            for id, arg in enumerate(op.args):
                out += '\tmv\ta{}, {}\n'.format(id, reg.get_ssa_name(arg))
            syscall_num: IntegerAttr[Any] = cast(IntegerAttr[Any],
                                                 op.syscall_num)
            out += '\tli\ta7, {}\n'.format(syscall_num.value.data)
            out += '\tscall'
            if op.result and len(op.result.uses) > 0:
                out += '\n\tmv\t{}, a0'.format(reg.get_ssa_name(op.result))
        elif isinstance(op, ReturnOp):
            # don't print return ops
            continue
        else:
            out += "\t{}\t{}".format(name, ", ".join(get_all_regs(op)))

        if isinstance(
                op, Riscv2Rs1ImmOperation
                | Riscv1Rd1Rs1ImmOperation
                | Riscv1Rd2RsOperation) and op.comment is not None:
            out += f'\t\t# {op.comment.data}'

        out += "\n"

    return out


def run_riscv(code: str,
              extensions: List[Type[InstructionSet]] = [],
              unlimited_regs: bool = False,
              verbosity: int = 5):
    cfg = RunConfig(
        debug_instruction=False,
        verbosity=verbosity,
        debug_on_exception=False,
        unlimited_registers=unlimited_regs,
    )

    cpu = UserModeCPU([RV32I, RV32M, RV_Debug, *extensions], cfg)

    io = StringIO(code)

    loader = AssemblyFileLoader.instantiate(  # pyright: ignore[reportUnknownMemberType]
        'example.asm', {})
    assert isinstance(loader, AssemblyFileLoader)
    cpu.load_program(
        loader.parse_io(io))  # pyright: ignore[reportUnknownMemberType]

    try:
        mmu = cast(MMU, cpu.mmu)  # pyright: ignore[reportUnknownMemberType]
        cpu.launch(mmu.programs[-1], verbosity > 1)
    except Exception as ex:
        print(ex)
