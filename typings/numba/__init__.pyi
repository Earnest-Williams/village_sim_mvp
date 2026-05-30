from collections.abc import Callable
from typing import TypeVar, overload

_F = TypeVar("_F", bound=Callable[..., object])

@overload
def njit(func: _F, *, cache: bool = ..., fastmath: bool = ...) -> _F: ...
@overload
def njit(
    func: None = ...,
    *,
    cache: bool = ...,
    fastmath: bool = ...,
) -> Callable[[_F], _F]: ...
