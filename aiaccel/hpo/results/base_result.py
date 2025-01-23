class BaseResult:

    def __init__(self, filename_template: str) -> None:
        self.filename_template = filename_template

    def load(self) -> None:
        raise NotImplementedError
