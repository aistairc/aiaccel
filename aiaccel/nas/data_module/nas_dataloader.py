from abc import ABCMeta, abstractmethod


class NAS1shotDataLoader(metaclass=ABCMeta):
    """A base data loader class for 1 shot NAS."""

    @abstractmethod
    def get_supernet_train_dataloader(self):
        """Get a data loader to train supernet.

        Raises:
            NotImplementedError: Please implement this method.
        """
        raise NotImplementedError

    @abstractmethod
    def get_architecture_search_dataloader(self):
        """Get a data loader for the architecture search.

        Raises:
            NotImplementedError: Please implement this method.
        """
        raise NotImplementedError

    @abstractmethod
    def get_num_supernet_train_data(self):
        """Get a number of supernet train data.

        Raises:
            NotImplementedError: Please implement this method.
        """
        raise NotImplementedError

    @abstractmethod
    def get_num_architecture_search_data(self):
        """Get a number of architecture search data.

        Raises:
            NotImplementedError: Please implement this method.
        """
        raise NotImplementedError

    @abstractmethod
    def get_dims(self):
        """Get dimensions of images of the data loaders.

        Raises:
            NotImplementedError: Please implement this method.
        """
        raise NotImplementedError

    @abstractmethod
    def get_num_classes(self):
        """Get a number of classes of the data loaders.

        Raises:
            NotImplementedError: Please implement this method.
        """
        raise NotImplementedError
