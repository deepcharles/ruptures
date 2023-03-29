"""Helper function to check if two breakpoints list are comparable."""


class BadPartitions(Exception):
    """Exception raised when the partition is bad."""

    pass


def sanity_check(bkps1, bkps2):
    """Checks if two partitions are indeed partitions of the same signal.

    Args:
        bkps1 (list): list of the last index of each regime.
        bkps2 (list): list of the last index of each regime.

    Raises:
        BadPartitions: whenever a partition does not respect some conditions.

    Returns:
        None:
    """
    # checks if empty.
    for nom, bkps in zip(("first", "second"), (bkps1, bkps2)):
        if len(bkps) == 0:
            raise BadPartitions("The {} partition is empty.".format(nom))
    # checks if both ends with the same index.
    if max(bkps1) != max(bkps2):
        raise BadPartitions(
            "The end of the last regime is not the same for each of the "
            "partitions:\n{}\n{}".format(bkps1, bkps2)
        )
    # checks if there is repetition.
    for bkps in (bkps1, bkps2):
        seen = set()
        if any(i in seen or seen.add(i) for i in bkps):
            raise BadPartitions("Some indexes are repeated: {}".format(bkps))
