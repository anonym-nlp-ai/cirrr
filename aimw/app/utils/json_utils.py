import json
from typing import Any


class JSONProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def read_json_file(self) -> Any:
        """
        Reads a JSON file and returns its contents as a Python object.

        Returns:
            dict or list: The contents of the JSON file as a Python object.

        Raises:
            FileNotFoundError: If the file specified by `self.file_path` does not exist.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        with open(self.file_path, encoding="utf-8") as f:
            data = json.load(f)
        return data

    def split_and_save_chunks(
        self, chunk_size=1000, output_file_prefix="output", path="./"
    ):
        """
        Splits the data read from a JSON file into chunks of a specified size and saves each chunk into a separate JSON file.

        Args:
            chunk_size (int, optional): The size of each chunk. Defaults to 1000.
            output_file_prefix (str, optional): The prefix for the output file names. Defaults to 'output'.
            path (str, optional): The path to the directory where the output files will be saved. Defaults to './'.

        Returns:
            None

        Raises:
            FileNotFoundError: If the JSON file specified by `self.file_path` does not exist.
            json.JSONDecodeError: If the JSON file contains invalid JSON.

        Note:
            The output files will be saved in the specified directory with the format `{output_file_prefix}_{i}.json`, where `i` is the index of the chunk.
        """
        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than 0")
        if not self.file_path:
            raise FileNotFoundError("File path not specified or not found")
        data = self.read_json_file()
        chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

        for i, chunk in enumerate(chunks):
            with open(
                f"{path}{output_file_prefix}_{i}.json", "w", encoding="utf-8"
            ) as f:
                json.dump(chunk, f)
