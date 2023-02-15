from typing import ParamSpec, Callable, Concatenate, TypeVar

from dataclasses import dataclass, field

from xdsl.ir import Operation, OpResult, Attribute, Region, Block
from xdsl.dialects.builtin import FunctionType


@dataclass
class Builder:
    ops: list[Operation] = field(default_factory=list)

    def get_ops(self) -> list[Operation]:
        return self.ops

    def add_op(self, op: Operation):
        self.ops.append(op)


P = ParamSpec('P')


def foo_op_builder(
    func: Callable[P, Operation]
) -> Callable[Concatenate[Builder, P], tuple[OpResult, ...]]:

    def impl(builder: Builder, *args: P.args,
             **kwargs: P.kwargs) -> tuple[OpResult, ...]:
        op = func(*args, **kwargs)
        builder.add_op(op)
        return tuple(op.results)

    return impl


def foo_op_builder_0(
        func: Callable[P,
                       Operation]) -> Callable[Concatenate[Builder, P], None]:

    def impl(builder: Builder, *args: P.args, **kwargs: P.kwargs) -> None:
        op = func(*args, **kwargs)
        builder.add_op(op)

    return impl


def foo_op_builder_1(
    func: Callable[P,
                   Operation]) -> Callable[Concatenate[Builder, P], OpResult]:

    def impl(builder: Builder, *args: P.args, **kwargs: P.kwargs) -> OpResult:
        op = func(*args, **kwargs)
        builder.add_op(op)
        return op.results[0]

    return impl


def build_callable(
    input_types: list[Attribute], return_types: list[Attribute]
) -> Callable[[Callable[Concatenate[Builder, P], None]], tuple[Region,
                                                               FunctionType]]:

    def wrapper(
        func: Callable[Concatenate[Builder, P], None]
    ) -> tuple[Region, FunctionType]:

        def impl(*args: P.args, **kwargs: P.kwargs) -> list[Operation]:
            builder = Builder()

            func(builder, *args, **kwargs)

            return builder.get_ops()

        region = Region.from_block_list(
            [Block.from_callable(input_types, impl)])
        ftype = FunctionType.from_lists(input_types, return_types)
        return region, ftype

    return wrapper


T = TypeVar('T')


# ((R, F, ...) -> T) -> ((...) -> ((R, F) -> T))
def foo_func_op_builder(
    func: Callable[Concatenate[Region, FunctionType, P], T]
) -> Callable[P, Callable[[tuple[Region, FunctionType]], T]]:

    # (...) -> (R, F) -> T
    def wrapper(
            *args: P.args,
            **kwargs: P.kwargs) -> Callable[[tuple[Region, FunctionType]], T]:

        # (R, F) -> T
        def inner(region_and_type: tuple[Region, FunctionType]) -> T:
            region, ftype = region_and_type
            return func(region, ftype, *args, **kwargs)

        return inner

    return wrapper