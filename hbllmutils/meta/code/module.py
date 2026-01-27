import os.path
import re


def get_pythonpath_of_source_file(source_file: str):
    module_dir = os.path.normpath(os.path.abspath(os.path.dirname(source_file)))
    while os.path.exists(os.path.join(module_dir, '__init__.py')):
        module_dir = os.path.dirname(module_dir)

    rel_file = os.path.relpath(source_file, module_dir)
    segments_text, _ = os.path.splitext(rel_file)
    module_text = re.sub(r'[\\/]+', '.', segments_text)
    return module_dir, module_text
