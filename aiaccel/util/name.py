from __future__ import annotations

import logging
import string
from typing import Any

from numpy.random import RandomState


def generate_random_name(rng: RandomState, length: int = 10) -> str:
    """Generate random name using alphanumeric.

    Args:
        rng (RandomState): A reference to a random generator.
        length (int, optional): A length of the name. This value should be
            greater than 0. Defaults to 10.

    Raises:
        ValueError: Occurs if the specified length is less than 1.

    Returns:
        str: A generated name.
    """

    if length < 1:
        logger = logging.getLogger('root.optimizer.name')
        logger.error('Name length should be greater than 0.')
        raise ValueError('Name length should be greater than 0.')

    rands = [
        rng.choice(list(string.ascii_letters + string.digits))[0]
        for _ in range(length)
    ]

    return ''.join(rands)
