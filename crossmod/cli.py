import argparse
import json
from pathlib import Path

from .config.loader import load_config

TASK_SCRIPTS = {
    'close_drawer': 'close_drawer.py',
    'insert_square_peg': 'insert_square_peg.py',
    'lift_numbered_block': 'lift_numbered_block.py',
    'play_jenga': 'play_jenga.py',
    'push_button': 'push_button_task.py',
    'rubish_in_bin': 'rubish_in_bin_task.py',
    'slide_block_to_target': 'slide_block_to_target.py',
}


def main(argv=None):
    parser = argparse.ArgumentParser(prog='crossmod', description='CrossMod CLI')
    sub = parser.add_subparsers(dest='cmd', required=True)

    run_task = sub.add_parser('run-task', help='Run a task script with config')
    run_task.add_argument('--task', required=True, choices=TASK_SCRIPTS.keys())
    run_task.add_argument('--config', required=False, help='Path to JSON/YAML config')
    run_task.add_argument('--extra', nargs='*', default=[], help='key=value overrides')

    args = parser.parse_args(argv)

    if args.cmd == 'run-task':
        cfg = load_config(args.config, overrides=args.extra)
        script = TASK_SCRIPTS[args.task]
        # For now, invoke the script by importing it so existing entrypoints remain intact
        # The scripts can read cfg from a conventional location if they choose.
        cfg_path = Path('.crossmod_runtime_config.json')
        cfg_path.write_text(json.dumps(cfg, indent=2))
        print(f'[crossmod] Wrote runtime config to {cfg_path.resolve()}')
        print(f'[crossmod] Importing and running {script} (ensure script reads runtime config if needed).')
        # Defer actual execution flow to the script when refactor lands.
        __import__(Path(script).stem)


if __name__ == '__main__':
    main()
