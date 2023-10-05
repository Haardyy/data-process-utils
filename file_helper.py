import os
import pandas


def create_pandas_dataframe_of_files(root_folder: str, file_name_contains: str = None):
    data_frame = pandas.DataFrame(get_dict_of_files_cycle(root_folder=root_folder))

    if file_name_contains is not None:
        data_frame = data_frame.loc[data_frame['file_name'].str.contains(file_name_contains)]

    data_frame = data_frame.sort_values(by=['depth', 'file_name', 'path_without_root_and_filename'])
    data_frame = data_frame.reset_index(drop=True)
    return data_frame


def get_dict_of_files_cycle(root_folder: str, depth=0, original_root: str = None):
    if original_root is None:
        original_root = root_folder

    files_and_folders = os.listdir(root_folder)
    ret = {'depth': list(), 'file_name': list(), 'path_without_root_and_filename': list(), 'path_with_root': list()}

    for item in files_and_folders:
        full_path = os.path.join(root_folder, item)

        if os.path.isdir(full_path):
            nested_ret = get_dict_of_files_cycle(full_path, depth=depth+1, original_root=original_root)
            ret['depth'] = ret['depth'] + nested_ret['depth']
            ret['file_name'] = ret['file_name'] + nested_ret['file_name']
            ret['path_without_root_and_filename'] = ret['path_without_root_and_filename'] + nested_ret[
                'path_without_root_and_filename']
            ret['path_with_root'] = ret['path_with_root'] + nested_ret['path_with_root']
        else:
            ret['depth'].append(depth)
            ret['file_name'].append(item)
            ret['path_without_root_and_filename'].append(
                unify_separators_in_path(full_path[len(original_root):len(full_path) - len(item)]))
            ret['path_with_root'].append(unify_separators_in_path(full_path))

    return ret


def unify_separators_in_path(path: str, use_separator=os.sep):
    if use_separator == '/':
        return path.replace("\\", '/')
    elif use_separator == "\\":
        return path.replace('/', '\\')
    else:
        raise Exception("Unexpected separator type - " + use_separator)


def get_directory_separator(path):
    if "/" in path and "\\" in path:
        raise Exception("Path contains both directory separators!")

    if "/" in path:
        return "/"
    elif "\\" in path:
        return "\\"
    else:
        raise Exception("Path did not contains directory separator!")


def check_directories_in_path(path, custom_text=None, stop_code=True):
    if path[-1] != '/':
        path += '/'

    if not os.path.exists(path):
        os.makedirs(path)
    else:
        if custom_text is None:
            custom_text = "Target FOLDER already exist! Do u want remove folder with files? [y]: "

        if input(custom_text) == 'y':
            files_in_dir = os.listdir(path)
            for file in files_in_dir:
                os.remove(f'{path}/{file}')
        else:
            print("Folder wasn't removed...")
            if stop_code:
                raise Exception("STOPPED EXECUTION!")


def remove_files(paths_to_files):
    for file in paths_to_files:
        os.remove(file)
