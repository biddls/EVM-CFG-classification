from typing import Generator
import numpy as np
from CFG_reader import CFG_Reader
from glob import glob
from dataclasses import dataclass, field
from avaliableOpCodes import AvaliableOpCodes, specialTokens
import collections
import numpy.typing as npt
from tqdm import tqdm
from collections import Counter

@dataclass
class CFG_Loader:
    """
    Loads all the CFG's dynamicaly to avoid having to load them all at once
    """
    avaliable_cfgs: dict[str, str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        files = glob("./src/ControlFlowGraphs/evmOut/*.json")

        # self.avaliable_cfgs = {addr: file path}
        for file in files:
            self.avaliable_cfgs[file.split("\\")[-1].split(".")[0]] = file

    # dict get item method
    def __getitem__(self, key: str) -> CFG_Reader:
        return CFG_Reader(self.avaliable_cfgs[key])

    # dict iter method
    def __iter__(self) -> Generator[CFG_Reader, None, None]:
        addrs = list(self.avaliable_cfgs.keys())
        for addr in addrs:
            yield self[addr]

    def __len__(self) -> int:
        return len(self.avaliable_cfgs.keys())


class Tokeniser:
    """
    Tokenises the opcodes in the CFG
    Tokeniseation:
    OPCODES
    values:
    <ZERO_ADDRESS>
    <MAX_ADDRESS>
    <ADDRESS>
    <ZERO>
    <MAX>
    <NUMBER_1-20>
    """
    addrLen = 40
    ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"
    MAX_ADDRESS = "0xffffffffffffffffffffffffffffffffffffffff"

    @staticmethod
    def preProcessing(cfg: CFG_Reader) -> list[list[str | tuple[str, str]]]:
        """
        Tokenises the opcodes in the CFG
        """
        tokens: list[list[str | tuple[str, str]]] = list()
        # splits the opcodes into tokens
        for node in cfg.parsedOpcodes[:-1]:
            nodeTokens = list()
            for opcode in node:
                if opcode == "EXIT BLOCK":
                    continue
                elif " " in opcode:
                    opcode = opcode.split(" ")
                    opcode = tuple(opcode)
                    if len(opcode) == 2:
                        nodeTokens.append(opcode)
                else:
                    if opcode == "INVALID":
                        continue
                    nodeTokens.append(opcode)
            tokens.append(nodeTokens)
        tokens.append(["EXIT BLOCK"])
        # Gives me something that looks like this for each node:
        # ['ADD', 'MSTORE', 'ADD', ['PUSH2', '0x5ef0'], 'JUMP', 'EXIT BLOCK']
        # print("###")
        # print(tokens[-2:])
        # print("###")
        return tokens

    @staticmethod
    def tokenise(tokens: list[list[str | tuple[str, str]]]) -> tuple[list[npt.NDArray[np.bool_]], list[int]]:
        """
        performs the tokeniseation for each node in the CFG
        returns a list of matrices for each node
        """
        vectors = list()
        for node in filter(lambda node: len(node), tokens):
            temp = Tokeniser.tokeniseNode(node)
            temp = Tokeniser.oneHotEncodeNode(temp)
            temp = Tokeniser.vectoriseNode(temp)
            vectors.append(temp)

        return vectors, list()

        # todo: implement this
        counts = Counter(vectors)
        counts: dict[list[int | tuple[int, int]], int] = dict(counts)

        newVectors: list[npt.NDArray[np.bool_]] = list()
        for temp in counts.keys():
            temp = Tokeniser.vectoriseNode(temp)
            newVectors.append(temp)

        return newVectors, list(counts.values())

    @staticmethod
    def tokeniseNode(byteCodes: list[str | tuple[str, str]]) -> list[str | tuple[str, str]]:
        # Takes this and gives me the list of values
        # that should correspond to what tokens
        frequency = collections.Counter([
            token[1] for token in byteCodes if isinstance(token, tuple)
        ])
        frequency = dict(frequency)
        """
        Need to check for the following and assign accordingly:
            <ZERO>
            <MAX>
            <NUMBER_1-20>
            <ADDRESS>
            <ZERO_ADDRESS>
            <MAX_ADDRESS>
        """
        # if tokens == [["EXIT BLOCK"]]:
        #     print("EXIT BLOCK")
        #     exit(0)
        #     return ["EXIT BLOCK"]
        # else:
        #     print(tokens)
        encodings = dict()
        # Addresses
        addrFreq = dict()
        addrFreq = {
            key: value
            for key, value in frequency.items()
            if len(key[2:]) == Tokeniser.addrLen
        }
        for key in addrFreq:
            del frequency[key]
        # <ZERO_ADDRESS>
        if Tokeniser.ZERO_ADDRESS in addrFreq:
            encodings[Tokeniser.ZERO_ADDRESS] = "<ZERO_ADDRESS>"
            del addrFreq[Tokeniser.ZERO_ADDRESS]

        # <MAX_ADDRESS>
        if Tokeniser.MAX_ADDRESS in addrFreq:
            encodings[Tokeniser.MAX_ADDRESS] = "<MAX_ADDRESS>"
            del addrFreq[Tokeniser.MAX_ADDRESS]

        # <ADDRESS>
        if addrFreq != {}:
            for key in addrFreq.keys():
                encodings[key] = "<ADDRESS>"

        numbersAssigned = 0

        for key in sorted(
            frequency.keys(), key=lambda item: item[1], reverse=True):
            shortKey = key[2:]
            # <ZERO>
            if set(shortKey) == {"0"}:
                encodings[key] = "<ZERO>"
                del frequency[key]
            # <MAX>
            elif set(shortKey) == {"f"}:
                encodings[key] = "<MAX>"
                del frequency[key]

            # Checks for numbers
            # <NUMBER_1-20>
            elif numbersAssigned < 20:
                encodings[key] = f"<NUMBER_{numbersAssigned}>"
                del frequency[key]
                numbersAssigned += 1

        # Replaces the tokens with the encodings
        for i, token in enumerate(byteCodes):
            if isinstance(token, tuple):
                if token[1] in encodings:
                    byteCodes[i] = (token[0], encodings[token[1]])
                else:
                    byteCodes[i] = token[0]

        return byteCodes

    @staticmethod
    def oneHotEncodeNode(tokens: list[str | tuple[str, str]]) -> list[int|tuple[int, int]]:
        """
        One hot encodes the tokens into their index
        """
        # print(AvaliableOpCodes)
        vectors = list()
        for token in tokens:
            if isinstance(token, tuple):
                # convert to more fancy vector
                vectors.append(
                    (AvaliableOpCodes[token[0]], specialTokens[token[1]]))
            else:
                vectors.append(AvaliableOpCodes[token])

        return vectors

    @staticmethod
    def vectoriseNode(tokens: list[int | tuple[int, int]]) -> npt.NDArray[np.bool_]:
        """
        Vectorises the tokens into a 2D numpy array
        """
        # define the shape of the array
        width = len(AvaliableOpCodes) + len(specialTokens)
        height = len(tokens)
        # create the array of 0's
        vector = np.zeros((height, width), dtype=bool)

        for i, token in enumerate(tokens):
            if isinstance(token, tuple):
                # convert to more fancy vector
                vector[i, token[0]] = True
                vector[i, len(AvaliableOpCodes) + token[1]] = True
            else:
                vector[i, token] = True

        return vector


if __name__ == "__main__":
    loader = CFG_Loader()
    tokens = list()
    lengths = list()
    for cfg in tqdm(loader):
        tokens = Tokeniser.preProcessing(cfg)
        vectors = Tokeniser.tokenise(tokens)
        for vector in vectors:
            lengths.append(vector.shape[0])
    print(max(lengths))
    print(min(lengths))
    print(sum(lengths)/len(lengths))
    print(np.median(lengths))
    
    import matplotlib.pyplot as plt
    plt.hist(lengths, bins=200)
    plt.show()
