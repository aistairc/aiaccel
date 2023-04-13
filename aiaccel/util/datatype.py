class DataType:
    """ Data type for hyperparameter optimization.
    """
    def is_uniform_int(self, data_type: str) -> bool:
        return data_type.lower() == 'uniform_int'

    def is_uniform_float(self, data_type: str) -> bool:
        return data_type.lower() == 'uniform_float'

    def is_categorical(self, data_type: str) -> bool:
        return data_type.lower() == 'categorical'

    def is_ordinal(self, data_type: str) -> bool:
        return data_type.lower() == 'ordinal'
