from typing import TYPE_CHECKING, List

from slitheryn.core.declarations import Enum
from slitheryn.core.declarations.top_level import TopLevel

if TYPE_CHECKING:
    from slitheryn.core.scope.scope import FileScope


class EnumTopLevel(Enum, TopLevel):
    def __init__(
        self, name: str, canonical_name: str, values: List[str], scope: "FileScope"
    ) -> None:
        super().__init__(name, canonical_name, values)
        self.file_scope: "FileScope" = scope
