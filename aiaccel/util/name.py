from typing import Union
import logging
import random
import string


def generate_random_name(length: int = 10) -> Union[str, None]:
    """
    Generate random name using alphanumeric.

    Args:
        length(int): A length of the name

    Returns:
        (Union[str, None]): A generated name.
    """

    if length < 1:
        logger = logging.getLogger('root.optimizer.name')
        logger.error('Name length should be greater than 0.')

        return None

    rands = [random.choice(string.ascii_letters + string.digits)
             for _ in range(length)]

    return ''.join(rands)
