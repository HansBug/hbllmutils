import io

from .module import get_package_name
from .source import get_source_info


def get_prompt_for_source_file(source_file: str, level: int = 2):
    """
    Generate a comprehensive code prompt for LLM analysis.

    This function creates a structured prompt containing source code and its dependencies,
    suitable for various LLM tasks like documentation generation, unit testing, or code analysis.
    """
    source_info = get_source_info(source_file)

    with io.StringIO() as sf:
        # Main source code section
        print(f'{"#" * level} Primary Source Code Analysis', file=sf)
        print(f'', file=sf)
        print(f'**Source File Location:** `{source_info.source_file}`', file=sf)
        print(f'', file=sf)
        print(f'**Package Namespace:** `{source_info.package_name}`', file=sf)
        print(f'', file=sf)
        print(f'**Complete Source Code:**', file=sf)
        print(f'', file=sf)
        print(f'```python', file=sf)
        print(source_info.source_code, file=sf)
        print(f'```', file=sf)
        print(f'', file=sf)

        # Import dependencies section
        if source_info.imports:
            print(f'{"#" * level} Dependency Analysis - Import Statements and Their Implementations', file=sf)
            print(f'', file=sf)
            print(f'The following section contains all imported dependencies for package `{source_info.package_name}` '
                  f'along with their source code implementations. This information can be used as reference context '
                  f'for understanding the main code\'s functionality and dependencies.', file=sf)
            print(f'', file=sf)

            for imp in source_info.imports:
                print(f'{"#" * (level + 1)} Import: `{imp.statement}`', file=sf)
                print(f'', file=sf)

                # Source file information
                if imp.inspect.source_file:
                    print(f'**Source File:** `{imp.inspect.source_file}`', file=sf)
                    print(f'', file=sf)
                    print(f'**Full Package Path:** `{get_package_name(imp.inspect.source_file)}.{imp.statement.name}`',
                          file=sf)
                    print(f'', file=sf)

                # Source code or object representation
                if imp.inspect.has_source:
                    print(f'**Implementation Source Code:**', file=sf)
                    print(f'', file=sf)
                    print(f'```python', file=sf)
                    print(imp.inspect.source_code, file=sf)
                    print(f'```', file=sf)
                    print(f'', file=sf)
                else:
                    print(
                        f'**Note:** Source code is not available through Python\'s inspection mechanism. Below is the object representation:',
                        file=sf)
                    print(f'', file=sf)
                    print(f'```', file=sf)
                    print(imp.inspect.object, file=sf)
                    print(f'```', file=sf)
                    print(f'', file=sf)

        return sf.getvalue()
