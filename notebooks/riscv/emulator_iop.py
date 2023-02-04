from __future__ import annotations

from dataclasses import dataclass

from itertools import chain

from riscemu import RunConfig, UserModeCPU, RV32I, RV32M, AssemblyFileLoader
from riscemu.instructions import InstructionSet, Instruction

from io import StringIO
from typing import Any, List, Type, cast

from xdsl.ir import Operation
from xdsl.dialects.builtin import ModuleOp
from .riscv_ssa import (FuncOp, SSAValue, LabelAttr, IntegerAttr, DirectiveOp,
                        LabelOp, RiscvNoParamsOperation, ECALLOp, ReturnOp,
                        DataSectionOp, LIOp)

SCALL_EXIT = 93


class RV_Debug(InstructionSet):
    # this instruction will dissappear into our emualtor soon-ish
    def instruction_print(self, ins: Instruction):
        reg = ins.get_reg(0)
        print("register {} contains value {}".format(reg, self.regs.get(reg)))


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


def print_riscv_ssa(module: ModuleOp, memory: int = 1024) -> str:
    data_ops: list[Operation] = []
    text_ops: list[Operation] = []

    for op in module.regions[0].blocks[0].ops:
        if isinstance(op, DataSectionOp):
            # There should be only one data section
            data_ops = op.regions[0].blocks[0].ops
            continue

        assert isinstance(op, FuncOp)

        func_name = op.func_name.data
        func_ops = op.func_body.blocks[0].ops

        # add support for other functions in the future, printer assumes only main
        assert func_name == 'main'
        # The verifier should have caught this but the last op in a FuncOp should
        # be a ReturnOp
        assert isinstance(func_ops[-1], ReturnOp)

        text_ops.append(LabelOp.get(func_name))
        text_ops.extend(func_ops[:-1])

    preamble_ops = (
        DirectiveOp.get(".bss", ""),  # bss stands for block starting symbol
        LabelOp.get("heap"),
        DirectiveOp.get(".space", f'{memory}'),
    )

    data_ops_prefix = (DirectiveOp.get(".data", ""), ) if len(data_ops) else ()

    text_ops_prefix = (DirectiveOp.get(".text", ""), )

    # perform the "exit" syscall, opcode 93
    text_ops_suffix = (ECALLOp.get(93), )

    ops = chain(preamble_ops, data_ops_prefix, data_ops, text_ops_prefix,
                text_ops, text_ops_suffix)

    out = ""
    reg = _SSAVALNamer()

    def get_all_regs(op):
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
        else:
            out += "\t{}\t{}".format(name, ", ".join(get_all_regs(op)))

        out += "\n"

    return out


def run_riscv(code: str,
              extensions: List[Type[InstructionSet]] = [],
              unlimited_regs: bool = False):
    cfg = RunConfig(
        debug_instruction=False,
        verbosity=5,
        debug_on_exception=False,
        unlimited_registers=unlimited_regs,
    )

    cpu = UserModeCPU((RV32I, RV32M, RV_Debug, *extensions), cfg)

    io = StringIO(code)

    loader = AssemblyFileLoader.instantiate('example.asm', [])
    cpu.load_program(loader.parse_io(io))

    try:
        cpu.launch(cpu.mmu.programs[-1], True)
    except Exception as ex:
        print(ex)
