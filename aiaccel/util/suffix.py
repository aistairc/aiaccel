import datetime


class Suffix:
    """ Create suffix with datetime.
    """

    def __init__(self) -> None:
        pass

    @classmethod
    def date(cls) -> str:
        """ Get the now datetime.

        Returns:
            str
        """
        t = datetime.datetime.today()
        suffix = (
            f"{str(t.year).zfill(4)}{str(t.month).zfill(2)}{str(t.day).zfill(2)}_"
            f"{str(t.hour).zfill(2)}{str(t.minute).zfill(2)}{str(t.second).zfill(2)}"
        )
        return suffix
