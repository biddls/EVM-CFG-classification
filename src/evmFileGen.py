import sqlite3
from typing import Any
from tqdm import tqdm


class ByteCodeIO:
    """
    Reads the bytecode from the database and writes it to a file
    """
    def __init__(self):
        self.sqliteConnection = sqlite3.connect("contStore.db")
        self.cursor = self.sqliteConnection.cursor()

    def __enter__(self):
        return self

    def getColumn(self, table: str, column: str) -> list[str] | list[Any]:
        sqlite_select_query = f"""SELECT {str(column)} from {str(table)}"""
        self.cursor.execute(sqlite_select_query)
        records = self.cursor.fetchall()
        return records

    def __exit__(self, exc_type, exc_value, traceback):
        if self.cursor:
            self.cursor.close()
            # logging.info("The SQLite cursor is closed")
        if self.sqliteConnection:
            self.sqliteConnection.close()
            # logging.info("The SQLite connection is closed")


if __name__ == "__main__":
    with ByteCodeIO() as db:
        output = db.getColumn("contracts", "address, byteCode")

    for addr, byteCode in tqdm(output):
        with open(f"./src/ControlFlowGraphs/evmIn/{addr}.evm", "w+") as f:
            f.write(byteCode[2:])
