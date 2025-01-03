"""
Open a interactive Matplotlib object and plot the file

Usage:
Drag & Drop a .pickle file into the python script
"""

import pickle
import sys
import mplcursors

args = sys.argv
if len(args) > 1:
    filename = args[1]
else:
    filename = "inv2.pkl"

figx = pickle.load(open(filename, 'rb'))
print(type(figx))

mplcursors.cursor()

figx.show()

misc = input()

pass