from calendar import c
from typing import Any, Generator

from numpy import average, median
from CFG_reader import CFG_Reader
from glob import glob
from dataclasses import dataclass, field
from avaliableOpCodes import AvaliableOpCodes
import collections
from tqdm import tqdm


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
    <ZERO>
    <MAX>
    <SMALL_NUMBER>
    <MEDIUM_NUMBER>
    <LARGE_NUMBER>
    <EXTA_LARGE_NUMBER>
    <NUMBER_1-20>
    <ADDRESS>
    <ZERO_ADDRESS>
    <MAX_ADDRESS>
    """
    addrLen = 40
    ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"
    MAX_ADDRESS = "0xffffffffffffffffffffffffffffffffffffffff"

    @staticmethod
    def tokenise(cfg: CFG_Reader) ->Any:
        """
        Tokenises the opcodes in the CFG
        """
        tokens: list[str|tuple[str, str]] = list()
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
            <SMALL_NUMBER>
            <MEDIUM_NUMBER>
            <LARGE_NUMBER>
            <EXTA_LARGE_NUMBER>
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
            for key, value in frequency.items() if len(key[2:]) == Tokeniser.addrLen
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
            for key, in addrFreq.keys():
                encodings[key] = "<ADDRESS>"

        for key, value in sorted(
            frequency.items(), key=lambda item: item[1], reverse=True
        ):
            # <ZERO>
            if int(key[2:], 16) == 0:
                encodings[key] = "<ZERO>"
                del frequency[key]
            # <MAX>
            elif set(key[2:]) == {"f"}:
                encodings[key] = "<MAX>"
                del frequency[key]

            # Checks for numbers
            # <NUMBER_1-20>
            

            # left over
            # <SMALL_NUMBER>
            elif len(key[2:]) <= 4:
                encodings[key] = "<SMALL_NUMBER>"
            # <MEDIUM_NUMBER>
            elif len(key[2:]) <= 8:
                encodings[key] = "<MEDIUM_NUMBER>"
            # <LARGE_NUMBER>
            elif len(key[2:]) <= 12:
                encodings[key] = "<LARGE_NUMBER>"
            # <EXTA_LARGE_NUMBER>
            elif len(key[2:]) > 12:
                encodings[key] = "<EXTA_LARGE_NUMBER>"

        return tokens


if __name__ == "__main__":
    loader = CFG_Loader()
    tokens = list()
    maxAddrs: dict[CFG_Reader, dict[str, int]] = dict()

    for cfg in tqdm(loader):
    # for cfg in loader:
        tokens = Tokeniser.tokenise(cfg)
        break
