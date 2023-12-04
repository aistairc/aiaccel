from abc import ABCMeta, abstractmethod

from torch import nn


class NASModule(nn.Module, metaclass=ABCMeta):
    """An abstract class for search space classes.

    Args:
        name (str): class name
        **kwargs:

    Attributes:
        name (str): class name
        active_op (None):
    """

    def __init__(self, name, **kwargs):
        super(NASModule, self).__init__()
        self.name = name
        self.active_op = None

    @abstractmethod
    def forward(self, X):
        """forward function

        Args:
            X (torch.Tensor): input data
        """
        pass

    @abstractmethod
    def build(self, config, **kwargs):
        """the method for creating supernet

        Args:
            config (str): config file
            **kwargs:
        """
        raise NotImplementedError

    @abstractmethod
    def select_active_op(self, structure_info, **kwargs):
        """the method for creating subnet in "active_op" """
        raise NotImplementedError

    @abstractmethod
    def enumerate_categorical_variables(self):
        """the method for creating categorical variable dictionary"""
        raise NotImplementedError

    @abstractmethod
    def fix_arc(self, **kwargs):
        """the method for deleting supernet instance"""
        raise NotImplementedError

    @abstractmethod
    def print_active_op(self, log_dir=None):
        """the method for printing information for subnet"""
        raise NotImplementedError

    @abstractmethod
    def get_param_num_list(self):
        """the method for getting list of parameter nums"""
        raise NotImplementedError
