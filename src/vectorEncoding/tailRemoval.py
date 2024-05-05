"""
This removes the long tail of vectors that are not used frequently
This requires the mapping between the
    index stored in the graph and
    the value stored in "counts"
"""
from CFG_reader import CFG_Reader
from collections import Counter
from icecream import ic

def shrinkCounts(
    counts: Counter[tuple[int | tuple[int, int]]],
    cfgs: list[CFG_Reader],
    length: int = 5000
    ) -> tuple[list[CFG_Reader], Counter[tuple[int | tuple[int, int]]]]:
    """
    Parameters
    ----------
    counts: Counter[tuple[int | tuple[int, int]]]
        The counts of the tokens
    cfgs: list[CFG_Reader]
        The list of CFGs
    length: int
        The length of the long tail to remove

    Returns
    -------
    tuple[list[CFG_Reader], Counter[tuple[int | tuple[int, int]]]]
        The updated CFGs and counts

    Description
    -----------
    Removes the long tail of vectors that are not used frequently
    This requires the mapping between the
        index stored in the graph and
        the value stored in "counts"
    """
    if len(counts) < length:
        return cfgs, counts

    # find the list of indexes that are used infrequently
    _counts = dict(counts)
    _countsSorted = sorted(_counts.items(), key=lambda item: item[1], reverse=True)
    _countsSorted, _countsToRemove = _countsSorted[:length], _countsSorted[length:]
    _countsSorted = dict(_countsSorted)
    _countsToRemove = dict(_countsToRemove)

    # delete the infrequent counts
    for key in _countsToRemove.keys():
        del _counts[key]

    # search for the new indexes
    # build mapping from old to new
    mapping: dict[int, int] = dict()
    lookup = {x: i for i, x in enumerate(_counts.keys())}
    for i, key in enumerate(_countsSorted.keys()):
        oldIndex = lookup[key]
        mapping[oldIndex] = i

    # remove them from the graphs
    # update the indexes in the graphs
    for cfg in cfgs:
        oldMapping: list[tuple[int, int]] = cfg.graph.nodes(data='extIndex') # type: ignore
        # tuple[nodeIndex, externalIndex]
        for node in oldMapping:
            try:
                if node[1] in mapping:
                    cfg.setIndex(mapping[node[1]], CFG_NodeIndex=node[0])
                else:
                    cfg.setIndex(-1, CFG_NodeIndex=node[0])
            except KeyError as e:
                print(f"KeyError: {node}")
                raise e

    # remove them from the counts
    return cfgs, Counter(_countsSorted)
