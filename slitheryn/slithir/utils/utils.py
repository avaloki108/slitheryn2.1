from typing import Union, Optional

from slitheryn.core.variables.local_variable import LocalVariable
from slitheryn.core.variables.state_variable import StateVariable

from slitheryn.core.declarations.solidity_variables import SolidityVariable
from slitheryn.core.variables.top_level_variable import TopLevelVariable

from slitheryn.slithir.variables.temporary import TemporaryVariable
from slitheryn.slithir.variables.constant import Constant
from slitheryn.slithir.variables.reference import ReferenceVariable
from slitheryn.slithir.variables.tuple import TupleVariable
from slitheryn.core.source_mapping.source_mapping import SourceMapping

RVALUE = Union[
    StateVariable,
    LocalVariable,
    TopLevelVariable,
    TemporaryVariable,
    Constant,
    SolidityVariable,
    ReferenceVariable,
]

LVALUE = Union[
    StateVariable,
    LocalVariable,
    TemporaryVariable,
    ReferenceVariable,
    TupleVariable,
]


def is_valid_rvalue(v: Optional[SourceMapping]) -> bool:
    return isinstance(
        v,
        (
            StateVariable,
            LocalVariable,
            TopLevelVariable,
            TemporaryVariable,
            Constant,
            SolidityVariable,
            ReferenceVariable,
        ),
    )


def is_valid_lvalue(v: Optional[SourceMapping]) -> bool:
    return isinstance(
        v,
        (
            StateVariable,
            LocalVariable,
            TemporaryVariable,
            ReferenceVariable,
            TupleVariable,
        ),
    )
