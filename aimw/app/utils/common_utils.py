from pathlib import Path
from typing import Union

from loguru import logger


def get_single_filepath_with_prefix(dir_path: str, prefix: str) -> Union[Path, None]:
    dir = Path(dir_path)

    for file in dir.iterdir():
        if file.name.startswith(prefix):
            logger.debug(f"{file}")
            return file
    return None


def get_single_filename_with_prefix(dir_path: str, prefix: str) -> Union[str, None]:

    dir = Path(dir_path)

    for file in dir.iterdir():
        if file.name.startswith(prefix):
            logger.debug(f"{file.name}")
            return file.name
    return None


def get_all_filepaths_with_prefix(
    dir_path: str, prefix: str
) -> Union[list[Path], None]:
    dir = Path(dir_path)
    file_paths = []

    for file in dir.iterdir():
        if file.name.startswith(prefix):
            file_paths.append(file)

    logger.debug(f"{file_paths}")

    return file_paths


def get_all_filenames_with_prefix(dir_path: str, prefix: str) -> Union[list[str], None]:
    dir = Path(dir_path)
    file_names = []

    for file in dir.iterdir():
        if file.name.startswith(prefix):

            file_names.append(file.name)
    logger.debug(f"{file_names}")
    return file_names


def get_all_filepaths(dir_path: str) -> Union[list[Path], None]:
    dir = Path(dir_path)
    file_paths = []

    for file in dir.iterdir():
        file_paths.append(file)

    logger.debug(f"{file_paths}")

    return file_paths


def get_all_filenames(dir_path: str) -> Union[list[str], None]:
    dir = Path(dir_path)
    file_names = []

    for file in dir.iterdir():
        file_names.append(file.name)

    logger.debug(f"{file_names}")

    return file_names


def get_exec_file_path(file_name: str) -> Union[str, None]:
    """the path for a given executable.

    Args:
        file_name (str): executable file name.

    Returns:
        Union[str, None]: the path of a given executable.
    """
    import shutil

    return shutil.which(file_name)


def write_markdown_file(content, filename):
    """Writes the given content as a markdown file to the local directory.

    Args:
      content: The string content to write to the file.
      filename: The filename to save the file as.
    """
    if type(content) == dict:
        content = "\n".join(f"{key}: {value}" for key, value in content.items())
    if type(content) == list:
        content = "\n".join(content)
    with open(f"./output/{filename}.md", "w") as f:
        f.write(content)
