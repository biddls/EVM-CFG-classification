import os
from glob import glob
from tqdm import tqdm

files = glob('./src/ControlFlowGraphs/evmIn/*.evm')

for fileAddr in tqdm(files):
    # gets the file name without the path and extension
    fileAddr = fileAddr.split('\\')[-1].split('.')[0]

    os.system(f'java -jar src/ControlFlowGraphs/EtherSolve.jar -r -j -o ./src/ControlFlowGraphs/evmOut/{fileAddr}.json ./src/ControlFlowGraphs/evmIn/{fileAddr}.evm')
