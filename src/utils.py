import skimage.morphology as morpho

##### MATHEMATICAL MORPHOLOGY


def strel(shape, size):
    """returns the chosen structuring element
     'diamond'  closed ball for the  L1 of radius size
     'disk'     closed ball for the  L2 of radius size
     'square'   square  of size size
    """

    if shape == 'diamond':
        return morpho.selem.diamond(size)
    if shape == 'disk':
        return morpho.selem.disk(size)
    if shape == 'square':
        return morpho.selem.square(size)

    raise RuntimeError('Erreur dans fonction strel: forme incomprise')