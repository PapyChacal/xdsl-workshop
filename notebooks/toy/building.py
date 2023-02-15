from typing import ParamSpec, Callable, Concatenate

from dataclasses import dataclass, field

from xdsl.ir import Operation, OpResult


@dataclass
class OpListBuilder:
    ops: list[Operation] = field(default_factory=list)

    def get_ops(self) -> list[Operation]:
        return self.ops

    def add_op(self, op: Operation):
        self.ops.append(op)


P = ParamSpec('P')


def foo_op_builder(
    func: Callable[P, Operation]
) -> Callable[Concatenate[OpListBuilder, P], tuple[OpResult, ...]]:

    def impl(builder: OpListBuilder, *args: P.args,
             **kwargs: P.kwargs) -> tuple[OpResult, ...]:
        op = func(*args, **kwargs)
        builder.add_op(op)
        return tuple(op.results)

    return impl


def foo_op_builder_0(
    func: Callable[P, Operation]
) -> Callable[Concatenate[OpListBuilder, P], None]:

    def impl(builder: OpListBuilder, *args: P.args,
             **kwargs: P.kwargs) -> None:
        op = func(*args, **kwargs)
        builder.add_op(op)

    return impl


def foo_op_builder_1(
    func: Callable[P, Operation]
) -> Callable[Concatenate[OpListBuilder, P], OpResult]:

    def impl(builder: OpListBuilder, *args: P.args,
             **kwargs: P.kwargs) -> OpResult:
        op = func(*args, **kwargs)
        builder.add_op(op)
        return op.results[0]

    return impl