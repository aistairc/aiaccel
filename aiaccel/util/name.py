from __future__ import annotations

import logging
import string
from typing import Any


def generate_random_name(
    length: int = 10,
    rng: Any = None
) -> str | None:
    """
    Generate random name using alphanumeric.

    Args:
        length (int): A length of the name.
        rng (np.random.RandomState): A reference to a random generator.

    Returns:
        str | None: A generated name.
    """

    if length < 1:
        logger = logging.getLogger('root.optimizer.name')
        logger.error('Name length should be greater than 0.')

        return None

    rands = [
        rng.choice(list(string.ascii_letters + string.digits))[0]
        for _ in range(length)
    ]

    return ''.join(rands)
