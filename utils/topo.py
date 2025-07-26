def interior(Base, B):
    """
    Computes the intersection of the sets in Base with set B.

    Parameters:
    - Base: List of lists or sets, the base sets.
    - B: List or set, the set to intersect with.

    Returns:
    - interior: The intersection of Base sets with B.
    """
    interior = set()
    B_set = set(B)

    for subset in Base:
        if set(subset).issubset(B_set):
            interior.update(subset)

    return list(interior)

def closure(Base, B, A):
    """
    Computes the closure of set B with respect to Base and A.

    Parameters:
    - Base: List of lists or sets, the base sets.
    - B: List or set, the set to compute closure for.
    - A: List or set, the universal set.

    Returns:
    - closure: The closure of set B.
    """
    closed = [A.copy()]

    for subset in Base:
        Aux = A.copy()
        for elem in subset:
            if elem in Aux:
                Aux.remove(elem)
        closed.append(Aux)

    Aux = []
    B_set = set(B)

    for subset in closed:
        if B_set.issubset(set(subset)):
            Aux.append(subset)

    if len(Aux) > 1:
        closure = list(set.intersection(*map(set, Aux)))
    elif len(Aux) == 1:
        closure = Aux[0]
    else:
        closure = []

    return closure