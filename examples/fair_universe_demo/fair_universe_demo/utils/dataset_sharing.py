from logging import Logger
from multiprocessing.managers import BaseManager

from fair_universe_demo.utils.dataset import Data
from fair_universe_demo.utils.selection import load_train_set_data

logger = Logger("dataset")


PORT = 50505
KEY = b"fair_universe"


class DataManager(BaseManager):
    pass


class DataStore:
    def __init__(self) -> None:
        self._data: Data | None = None

    def get_or_load(self, root_dir: str) -> Data:
        if not self._data:
            self._data = load_train_set_data(root_dir)
        return self._data


def host_dataset(
    port: int = PORT,
    key: bytes = KEY,
) -> None:
    """Store the dataset in memory as long as the server is running

    Args:
        port (int, optional): Defaults to PORT.
        key (bytes, optional): Defaults to KEY.
    """
    store = DataStore()

    DataManager.register("get_store", callable=lambda: store)

    manager = DataManager(address=("", port), authkey=key)
    server = manager.get_server()

    print(f"FAIR Universe Dataset cached at {port=} and key={key.decode()}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shut down FAIR Universe Dataset and free up memory")


def fetch_dataset(
    root_dir: str,
    port: int = PORT,
    key: bytes = KEY,
) -> Data:
    """Access the Data object by first querying the server. If the server is not running it will
    attempt instead to load the data manually.

    Args:
        root_dir (str): Path to the FAIR Universe Data directory
        port (int, optional): Defaults to PORT.
        key (bytes, optional): Defaults to KEY.

    Returns:
        Data: Instance of the Data object with all the train and test rows loaded.
    """
    DataManager.register("get_store")
    manager = DataManager(address=("localhost", port), authkey=key)

    try:
        manager.connect()
        store = manager.get_store()  # type: ignore
        return store.get_or_load(root_dir)
    except ConnectionRefusedError:
        logger.info(f"FAIR Universe server not running. Use {__file__} to start it. Defaulting to reading from disk")
        return load_train_set_data(root_dir)


if __name__ == "__main__":
    host_dataset()
