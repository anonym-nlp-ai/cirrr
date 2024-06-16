class BaseException(Exception):
    def __init__(self, name: str):
        self.name = name


class ConfigException(BaseException):
    pass


class EmptyConfigException(BaseException):
    def __init__(self, param_name: str, message: str = "parameter is not defined."):
        self.param_name = param_name
        self.message = message
        super().__init__(message)

    def __str__(self):
        return f"'{self.param_name}' {self.message}"


class ValueConfigException(BaseException):
    def __init__(self, param_name: str, message: str = "parameter is not defined."):
        self.param_name = param_name
        self.message = message
        super().__init__(message)

    def __str__(self):
        return f"'{self.param_name}' {self.message}"


class ValueTypeException(BaseException):
    def __init__(self, param_name: str, expected_type, actual_type):
        self.param_name = param_name
        self.message = (
            f"Expected value of type {expected_type}, got value of type {actual_type}"
        )
        super().__init__(self.message)

    def __str__(self):
        return f"'{self.param_name}' {self.message}"
