from src import CFG_reader


class Test_CFG_reader:
    def test_parsedOpcodes(self):
        reader = CFG_reader.CFG_Reader()
        assert len(reader.parsedOpcodes) > 1
