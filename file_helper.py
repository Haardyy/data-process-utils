import os
import pandas

from dataclasses import dataclass
from typing import List


@dataclass
class FilePathing:
    depth: int
    file_name: str
    path_with_root: str
    path_without_root_and_filename: str


def convert_list_to_pandas_dataframe(files_list: List[FilePathing]) -> pandas.DataFrame:
    """A simple function for conveting a list with FilePathing objects into a pandas dataframe."""

    if not isinstance(files_list, list):
        raise TypeError(f"Expected files_list to be of type list, got {type(files_list).__name__} instead.")

    if not all(isinstance(file, FilePathing) for file in files_list):
        raise ValueError("All elements in files_list must be instances of the FilePathing class.")

    return pandas.DataFrame([file.__dict__ for file in files_list])


def search_for_files(root_folder: str, file_name_contains: str = None) -> List[FilePathing]:
    """The function searches for all files from the defined root folder.

            Parameters
            ----------
            root_folder : numpy.ndarray
                The path from which the search is performed.
            file_name_contains : str
                Filters out files that do not contain the specified string.

            Returns
            -------
            files_list : list
                Returns an ordered list of FilePathing objects.
                The ordering was done by (depth, file_name, path_without_root_and_filename).

            Raises
            ------
            TypeError
                If file_name_contains is defined and is not string.
        """

    files_list = recursive_file_search(root_folder=root_folder)

    if file_name_contains and type(file_name_contains) != str:
        raise TypeError('Parameter file_name_contains is required to be string type or None!')

    if file_name_contains:
        files_list = [file for file in files_list if file_name_contains in file.file_name]

    files_list.sort(key=lambda file: (file.depth, file.file_name, file.path_without_root_and_filename))

    return files_list


def recursive_file_search(root_folder: str, depth=0, original_root: str = None) -> List[FilePathing]:
    """Primary recursive function for the search_for_files function. It should not be used externally."""

    if original_root is None:
        original_root = root_folder

    files_and_folders = os.listdir(root_folder)
    ret = list()

    for item in files_and_folders:
        full_path = os.path.join(root_folder, item)

        if os.path.isdir(full_path):
            nested_ret = recursive_file_search(full_path, depth=depth+1, original_root=original_root)
            ret = ret + nested_ret

        else:
            ret.append(FilePathing(depth=depth,
                                   file_name=item,
                                   path_without_root_and_filename=unify_separators_in_path(
                                       path=full_path[len(original_root):len(full_path) - len(item)]),
                                   path_with_root=unify_separators_in_path(full_path)))

    return ret


def unify_separators_in_path(path: str, use_separator=os.sep):
    if use_separator == '/':
        return path.replace("\\", '/')
    elif use_separator == "\\":
        return path.replace('/', '\\')
    else:
        raise ValueError("Unexpected separator type - " + use_separator)


def get_directory_separator(path):
    if "/" in path and "\\" in path:
        raise ValueError("Path contains both directory separators!")

    if "/" in path:
        return "/"
    elif "\\" in path:
        return "\\"
    else:
        raise ValueError("Path did not contains directory separator!")
