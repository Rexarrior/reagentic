import argparse
import importlib
import sys


def main():
    parser = argparse.ArgumentParser(description='Run reagentic scripts as modules.')
    parser.add_argument('--script', help='Name of the script to run (e.g., changelog_agent).')

    args, unknown = parser.parse_known_args()

    if args.script:
        script_name = args.script
        module_name = f'reagentic.scripts.{script_name}'

        # try:
        module = importlib.import_module(module_name)
        if hasattr(module, 'main') and callable(module.main):
            module.main()
        else:
            print(f"Error: Script '{script_name}' found, but does not have a callable 'main' function.")
            sys.exit(1)
        # except ImportError:
        #     print(f"Error: Script '{script_name}' not found in the 'reagentic.scripts' package.")
        #     sys.exit(1)
        # except Exception as e:
        #     print(f"Error running script '{script_name}': {e}")
        #     sys.exit(1)
    else:
        print('No script specified. Use --script <script_name> to run a script.')
        sys.exit(1)


if __name__ == '__main__':
    main()
