from slitheryn.exceptions import SlitherException


class ParsingError(SlitherException):
    pass


class VariableNotFound(SlitherException):
    pass
