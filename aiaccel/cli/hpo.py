#!/usr/bin/env python3
"""HPO apps CLI for aiaccel."""

import argparse
import sys
from importlib import import_module


def main() -> None:
    """Main entry point for aiaccel-hpo."""
    parser = argparse.ArgumentParser(
        prog="aiaccel-hpo",
        description="Hyperparameter optimization apps for aiaccel",
    )
    parser.add_argument("command", help="Command to run (e.g., optimize)")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments to pass to the command")

    args = parser.parse_args()

    # Map of available commands to their module paths
    command_map = {
        "optimize": "aiaccel.hpo.apps.optimize",
    }

    if args.command not in command_map:
        print(f"Error: Unknown command '{args.command}'", file=sys.stderr)
        print(f"Available commands: {', '.join(command_map.keys())}", file=sys.stderr)
        sys.exit(1)

    # Import and run the module
    try:
        module_path = command_map[args.command]
        module = import_module(module_path)
        
        # Replace sys.argv to pass the remaining arguments to the target module
        original_argv = sys.argv[:]
        sys.argv = [f"aiaccel-hpo {args.command}"] + args.args
        
        try:
            module.main()
        finally:
            # Restore original sys.argv
            sys.argv = original_argv
            
    except ImportError as e:
        print(f"Error: Could not import module '{module_path}': {e}", file=sys.stderr)
        sys.exit(1)
    except AttributeError:
        print(f"Error: Module '{module_path}' does not have a main() function", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()