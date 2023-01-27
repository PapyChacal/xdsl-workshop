from riscemu import RunConfig, UserModeCPU, RV32I, RV32M, AssemblyFileLoader
from riscemu.instructions import InstructionSet, Instruction

from io import StringIO
from typing import List, Type
from xdsl.dialects.builtin import ModuleOp
from .riscv_ssa import *

SCALL_EXIT = 93


class _SSAVALNamer:

    ssa_val_names = {}
    idx = 0

    def __init__(self) -> None:
        self.ssa_val_names = dict()
        self.idx = 0

    def get_ssa_name(self, reg):
        if reg not in self.ssa_val_names:
            id = self.idx
            self.idx += 1
            self.ssa_val_names[reg] = id
        return str('%' + str(self.ssa_val_names[reg]))


def print_riscv_ssa(module: ModuleOp):
    out = ".text\n"
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

    for op in module.regions[0].ops:
        if isinstance(op, DirectiveOp):
            out += "{} {}".format(op.directive.data, op.value.data)
        elif isinstance(op, LabelOp):
            out += "{}:".format(op.label.data)
        elif isinstance(op, RiscvNoParamsOperation):
            out += "{}".format(op.name.split(".")[-1])
        elif isinstance(op, ECALLOp):
            for id, arg in enumerate(op.args):
                out += '\tmv\ta{}, {}\n'.format(id, reg.get_ssa_name(arg))
            out += '\tli\ta7, {}\n'.format(op.syscall_num.value.data)
            out += '\tscall'
        else:
            out += "\t{}\t{}".format(
                op.name.split(".")[-1], ", ".join(get_all_regs(op)))

        out += "\n"
    return out


def run_riscv(code: str, extensions: List[Type[InstructionSet]] = [], unlimited_regs=False):
    cfg = RunConfig(
        debug_instruction=False,
        verbosity=5,
        debug_on_exception=False,
        unlimited_registers=unlimited_regs,
    )

    cpu = UserModeCPU((RV32I, RV32M) + extensions, cfg)

    io = StringIO(code)

    loader = AssemblyFileLoader.instantiate('example.asm', [])
    cpu.load_program(loader.parse_io(io))

    try:
        cpu.launch(cpu.mmu.programs[-1], True)
    except Exception as ex:
        print(ex)
