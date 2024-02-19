"""
This performs the mapping
    from the generated CFG vectorx to
    the corolarry node in the CFG
"""
import numpy as np
from CFG_reader import CFG_Reader
from tokeniser import Tokeniser, CFG_Loader
from tqdm import tqdm
from collections import Counter

class CFG_Vec_reconst:
    ...