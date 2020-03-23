#!/usr/bin/env python3
'''
This file changes `warning`'s warn format so a) the message is only printed
once, and b) the line number and file location of the offending line is printed.
Credit to https://pymotw.com/2/warnings/
'''

import warnings

def one_line_warning(message, category, filename, lineno, file=None, line=None):
    return f"{filename}:{lineno}: {category.__name__}:{message}"

warnings.formatwarning = one_line_warning
