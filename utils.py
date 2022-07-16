
def progressbar(percentage, endstr='', barlenth=20):
    if int(percentage)==1: endstr +='\n'
    print('\r[' + '>' * int(percentage * barlenth) +
          '-' * (barlenth - int(percentage * barlenth)) + ']',
          format(percentage * 100, '.1f'), '%', end=' '+endstr)