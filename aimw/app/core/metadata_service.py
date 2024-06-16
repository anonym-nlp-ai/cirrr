import pathlib

import tomli

toml_file_path = str(pathlib.Path(
    __file__).resolve().parent.parent.parent / "pyproject.toml")


def _get_api_meta() -> dict:
    with open(toml_file_path, mode='rb') as pyproject:

        pkg_metadata = tomli.load(pyproject)['tool']['poetry']
        return pkg_metadata


def _get_api_version() -> str:
    pkg_meta = _get_api_meta()
    version_version = str(pkg_meta['version'])
    
    return version_version

def _get_api_name() -> str:
    pkg_meta = _get_api_meta()
    project_name = str(pkg_meta['name'])

    return project_name
