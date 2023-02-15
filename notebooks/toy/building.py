from __future__ import annotations

from typing import ParamSpec, Callable, Concatenate, TypeVar

from dataclasses import dataclass, field

from xdsl.ir import Operation, OpResult, Attribute, Region, Block
from xdsl.dialects.builtin import FunctionType

_P = ParamSpec('_P')
_T = TypeVar('_T')


@dataclass
class Builder:
    ops: list[Operation] = field(default_factory=list)

    def get_ops(self) -> list[Operation]:
        return self.ops

    def add_op(self, op: Operation):
        self.ops.append(op)

    @staticmethod
    def build_op_list(func: Callable[[Builder], None]) -> list[Operation]:

        builder = Builder()

        func(builder)

        return builder.get_ops()


def foo_op_builder(
    func: Callable[_P, Operation]
) -> Callable[Concatenate[Builder, _P], tuple[OpResult, ...]]:

    def impl(builder: Builder, *args: _P.args,
             **kwargs: _P.kwargs) -> tuple[OpResult, ...]:
        op = func(*args, **kwargs)
        builder.add_op(op)
        return tuple(op.results)

    return impl


def foo_op_builder_0(
        func: Callable[_P,
                       Operation]) -> Callable[Concatenate[Builder, _P], None]:

    def impl(builder: Builder, *args: _P.args, **kwargs: _P.kwargs) -> None:
        op = func(*args, **kwargs)
        builder.add_op(op)

    return impl


def foo_op_builder_1(
    func: Callable[_P,
                   Operation]) -> Callable[Concatenate[Builder, _P], OpResult]:

    def impl(builder: Builder, *args: _P.args,
             **kwargs: _P.kwargs) -> OpResult:
        op = func(*args, **kwargs)
        builder.add_op(op)
        return op.results[0]

    return impl


def build_callable(
    input_types: list[Attribute], return_types: list[Attribute]
) -> Callable[[Callable[Concatenate[Builder, _P], None]], tuple[Region,
                                                                FunctionType]]:

    def wrapper(
        func: Callable[Concatenate[Builder, _P], None]
    ) -> tuple[Region, FunctionType]:

        def impl(*args: _P.args, **kwargs: _P.kwargs) -> list[Operation]:
            builder = Builder()

            func(builder, *args, **kwargs)

            return builder.get_ops()

        region = Region.from_block_list(
            [Block.from_callable(input_types, impl)])
        ftype = FunctionType.from_lists(input_types, return_types)
        return region, ftype

    return wrapper


# ((B, R, F, ...) -> T) -> ((B, ...) -> ((R, F) -> T))
def foo_func_op_builder(
    func: Callable[Concatenate[Builder, Region, FunctionType, _P], _T]
) -> Callable[Concatenate[Builder, _P], Callable[[tuple[Region, FunctionType]],
                                                 _T]]:

    # (B, ...) -> (R, F) -> T
    def wrapper(
            builder: Builder, *args: _P.args, **kwargs: _P.kwargs
    ) -> Callable[[tuple[Region, FunctionType]], _T]:

        # (R, F) -> T
        def inner(region_and_type: tuple[Region, FunctionType]) -> _T:
            region, ftype = region_and_type
            return func(builder, region, ftype, *args, **kwargs)

        return inner

    return wrapper