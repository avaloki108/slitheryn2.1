from typing import List


from slitheryn.core.variables.variable import Variable
from slitheryn.slithir.operations import Operation


class Nop(Operation):
    @property
    def read(self) -> List[Variable]:
        return []

    @property
    def used(self):
        return []

    def __str__(self):
        return "NOP"
