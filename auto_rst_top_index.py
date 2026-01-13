import argparse
import os

from natsort import natsorted


def main():
    parser = argparse.ArgumentParser(description='Auto create rst docs top index for project')
    parser.add_argument('-i', '--input_dir', required=True, help='Input python project directory')
    parser.add_argument('-o', '--output', required=True, help='Output rst doc top index file')
    args = parser.parse_args()

    rel_names = []
    for name in os.listdir(args.input_dir):
        item_path = os.path.join(args.input_dir, name)
        if (os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, '__init__.py'))) or \
                (os.path.isfile(item_path) and name.endswith('.py') and not name.startswith('__')):
            if name.endswith('.py'):
                rel_names.append(os.path.splitext(name)[0])
            else:
                rel_names.append(name)

    rel_names = natsorted(rel_names)
    with open(args.output, 'w') as f:
        print(f'.. toctree::', file=f)
        print(f'    :maxdepth: 2', file=f)
        print(f'    :caption: API Documentation', file=f)
        print(f'', file=f)
        for name in rel_names:
            if os.path.exists(os.path.join(args.input_dir, name, '__init__.py')):
                print(f'    api_doc/{name}/index', file=f)
            else:
                print(f'    api_doc/{name}', file=f)
        print(f'', file=f)


if __name__ == '__main__':
    main()
