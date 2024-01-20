from typing import Generator
import numpy as np
from CFG_reader import CFG_Reader
from glob import glob
from dataclasses import dataclass, field
from avaliableOpCodes import AvaliableOpCodes, specialTokens
import collections


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
    def tokenise(cfg: CFG_Reader) -> list[str | tuple[str, str]]:
        """
        Tokenises the opcodes in the CFG
        """
        tokens: list[str | tuple[str, str]] = list()
        # splits the opcodes into tokens
        for node in cfg.parsedOpcodes[:-1]:
            for opcode in node:
                if " " in opcode:
                    opcode = opcode.split(" ")
                    opcode = tuple(opcode)
                    if len(opcode) == 2:
                        tokens.append(opcode)
                else:
                    tokens.append(opcode)
        tokens.append("EXIT BLOCK")
        # Gives me something that looks like this:
        # ['ADD', 'MSTORE', 'ADD', ['PUSH2', '0x5ef0'], 'JUMP', 'EXIT BLOCK']

        # Takes this and gives me the list of values
        # that should correspond to what tokens
        frequency = collections.Counter([
            token[1] for token in tokens if isinstance(token, tuple)
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
        for i, token in enumerate(tokens):
            if isinstance(token, tuple):
                if token[1] in encodings:
                    tokens[i] = (token[0], encodings[token[1]])
                else:
                    tokens[i] = token[0]

        return tokens

    @staticmethod
    def oneHotEncode(tokens: list[str | tuple[str, str]]) -> list[int|tuple[int, int]]:
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
    def vectorise(tokens: list[int | tuple[int, int]]) -> np.array:
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
    from timeit import repeat
    loader = CFG_Loader()
    tokens = list()
    for cfg in loader:
        tokens = Tokeniser.tokenise(cfg)
        indexes = Tokeniser.oneHotEncode(tokens)
        vector = Tokeniser.vectorise(indexes)
        break
