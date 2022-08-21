def de_parallel(dict):
    findict = {}
    for key in dict:
        finkey = key[7:]
        findict[finkey] = dict[key]
    return findict