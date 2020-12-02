import pyscf.dft as dft
from pyscf.dft.libxc import parse_xc


def find_in_codes(code):

    for key in dft.libxc.XC_CODES:
        if dft.libxc.XC_CODES[key] == code:
            return key

def parse_xc_code(xc_code):
    """ Parse pycsf style xc_code, that can contain +,*-, and combinations
    of functionals """
    success = False
    cnt = 0
    codes = {}
    code = -999
    orig_keys = {k for k in dft.libxc.XC_KEYS}
    orig_codes = {key:val for key, val in dft.libxc.XC_CODES.items()}
    while(not success and cnt < 20):
        try:
            parsed = parse_xc(xc_code)
            success = True
        except KeyError as e:
            name = e.args[0].split()[-1]
            dft.libxc.XC_KEYS.add(name)
            dft.libxc.XC_CODES[e.args[0].split()[-1]] = code
            codes[code] = name
            code += 1
        cnt += 1

    pars, funcs = parsed

    for i, f in enumerate(funcs):
        code, weight = f
        if code in codes:
            name = codes[code]
        else:
            name = find_in_codes(code)
        funcs[i] = (name, weight)

    dft.libxc.XC_KEYS = orig_keys
    dft.libxc.XC_CODES = orig_codes
    return parsed

def find_max_level(parsed_xc):

    xc_levels = {'LDA':0, 'GGA':1, 'MGGA':2}
    parsed_xc = parsed_xc[1]
    highest_xc = 'LDA'
    highest_level = 0
    for xc in parsed_xc:
        l = xc[0].split('_')[0]
        if xc_levels[l] > highest_level:
            highest_xc = l
            highest_level = xc_levels[l]

    return highest_xc    
