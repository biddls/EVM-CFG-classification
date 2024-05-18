"""
Runs the CFG generation tool EtherSolve on all the EVM files in the folder
"""
import os
from glob import glob
from tqdm import tqdm
import subprocess

files = glob('./src/ControlFlowGraphs/evmIn/*.evm')

for fileAddr in tqdm(files):
    # gets the file name without the path and extension
    fileAddr = fileAddr.split('\\')[-1].split('.')[0]

    # checks if the file has already been processed
    if os.path.isfile(f"./src/ControlFlowGraphs/evmOut/{fileAddr}.json"):
        continue

    out = subprocess.run(
        f"java -jar src/ControlFlowGraphs/EtherSolve.jar -r -j -o ./src/ControlFlowGraphs/evmOut/{fileAddr}.json ./src/ControlFlowGraphs/evmIn/{fileAddr}.evm",
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
