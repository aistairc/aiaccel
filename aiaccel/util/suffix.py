import datetime


class Suffix:
    """ Create suffix with datetime.
    """
    def __init__(self):
        pass

    @classmethod
    def date(cls):
        """ Get the now datetime.

        Returns:
            None
        """
        t = datetime.datetime.today()
        suffix = (
            "{}{}{}_{}{}{}"
            .format(
                str(t.year).zfill(4),
                str(t.month).zfill(2),
                str(t.day).zfill(2),
                str(t.hour).zfill(2),
                str(t.minute).zfill(2),
                str(t.second).zfill(2)
            )
        )
        return suffix
