from riscemu import RunConfig, UserModeCPU, RV32I, RV32M, AssemblyFileLoader
from riscemu.instructions import InstructionSet, Instruction

from io import StringIO
from typing import List, Type
from xdsl.dialects.builtin import ModuleOp
from .riscv_ssa import SSAValue, LabelAttr, IntegerAttr, DirectiveOp, \
    LabelOp, RiscvNoParamsOperation, ECALLOp

SCALL_EXIT = 93


class RV_Debug(InstructionSet):
    # this instruction will dissappear into our emualtor soon-ish
    def instruction_print(self, ins: Instruction):
        reg = ins.get_reg(0)
        print("register {} contains value {}".format(reg, self.regs.get(reg)))


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
    out = ""
    has_region_definitions = False
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
        name = '.'.join(op.name.split(".")[1:])
        if isinstance(op, DirectiveOp):
            out += "{} {}".format(op.directive.data, op.value.data)
            if op.directive.data.startswith("."):
                has_region_definitions = True
        elif isinstance(op, LabelOp):
            out += "{}:".format(op.label.data)
        elif isinstance(op, RiscvNoParamsOperation):
            out += "{}".format(name)
        elif isinstance(op, ECALLOp):
            for id, arg in enumerate(op.args):
                out += '\tmv\ta{}, {}\n'.format(id, reg.get_ssa_name(arg))
            out += '\tli\ta7, {}\n'.format(op.syscall_num.value.data)
            out += '\tscall'
            if op.result and len(op.result.uses) > 0:
                out += '\n\tmv\t{}, a0'.format(reg.get_ssa_name(op.result))
        else:
            out += "\t{}\t{}".format(name, ", ".join(get_all_regs(op)))

        out += "\n"
    if not has_region_definitions:
        out = ".text\n"+out
    return out


def run_riscv(code: str, extensions: List[Type[InstructionSet]] = [], unlimited_regs=False):
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
