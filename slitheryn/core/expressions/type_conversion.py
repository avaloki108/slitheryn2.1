from typing import Union, TYPE_CHECKING

from slitheryn.core.expressions.expression import Expression
from slitheryn.core.solidity_types.type import Type

if TYPE_CHECKING:
    from slitheryn.core.expressions.call_expression import CallExpression
    from slitheryn.core.expressions.identifier import Identifier
    from slitheryn.core.expressions.literal import Literal
    from slitheryn.core.expressions.member_access import MemberAccess
    from slitheryn.core.solidity_types.elementary_type import ElementaryType
    from slitheryn.core.solidity_types.type_alias import TypeAliasContract
    from slitheryn.core.solidity_types.user_defined_type import UserDefinedType


class TypeConversion(Expression):
    def __init__(
        self,
        expression: Union[
            "MemberAccess", "Literal", "CallExpression", "TypeConversion", "Identifier"
        ],
        expression_type: Union["ElementaryType", "UserDefinedType", "TypeAliasContract"],
    ) -> None:
        super().__init__()
        assert isinstance(expression, Expression)
        assert isinstance(expression_type, Type)
        self._expression: Expression = expression
        self._type: Type = expression_type

    @property
    def type(self) -> Type:
        return self._type

    @type.setter
    def type(self, new_type: Type) -> None:
        self._type = new_type

    @property
    def expression(self) -> Expression:
        return self._expression

    def __str__(self) -> str:
        return str(self.type) + "(" + str(self.expression) + ")"
