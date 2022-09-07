from aiaccel.util import aiaccel

# Create a wrapper
run = aiaccel.Run()

# Optimization
y = run.execute_and_report("python original_main.py")
ys = run.execute("python original_main.py")
run.report(ys[0])
print(run.ys)


# from aiaccel.config import load_config
# from aiaccel.util.time_tools import get_time_now
# from aiaccel.wrapper_tools import save_result
# from pathlib import Path
# import aiaccel
# import argparse
# import subprocess


# def parse_result(res):
#     # This should be editted by the user!
#     line = res.split('\n')
#     return float(line[1])


# def main():
#     # Parse arguments
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-c', '--config', default='config.json')
#     parser.add_argument('-i', '--trial_id', type=str, required=True)
#     parser.add_argument('-x1', '--x1', type=float)
#     parser.add_argument('-x2', '--x2', type=float)
#     parser.add_argument('-x3', '--x3', type=float)
#     parser.add_argument('-x4', '--x4', type=float)
#     parser.add_argument('-x5', '--x5', type=float)
#     parser.add_argument('-x6', '--x6', type=float)
#     parser.add_argument('-x7', '--x7', type=float)
#     parser.add_argument('-x8', '--x8', type=float)
#     parser.add_argument('-x9', '--x9', type=float)
#     parser.add_argument('-x10', '--x10', type=float)
#     args = parser.parse_args()

#     # Load config
#     config = load_config(args.config)

#     # Generate commands
#     commands = [
#         'python',
#         'original_main.py',
#         '-x1={}'.format(args.x1),
#         '-x2={}'.format(args.x2),
#         '-x3={}'.format(args.x3),
#         '-x4={}'.format(args.x4),
#         '-x5={}'.format(args.x5),
#         '-x6={}'.format(args.x6),
#         '-x7={}'.format(args.x7),
#         '-x8={}'.format(args.x8),
#         '-x9={}'.format(args.x9),
#         '-x10={}'.format(args.x10)
#     ]

#     # Measure start time
#     start_time = get_time_now()

#     # Run in local
#     res = subprocess.run(commands, stdout=subprocess.PIPE)

#     # Measure end time
#     end_time = get_time_now()
#     res = res.stdout.decode('utf-8')

#     # Need to generate result to evaluate the optimizer.
#     # The result of this example is directly same.

#     # Parse the result
#     result = parse_result(res)

#     # Save the result
#     ws = Path(config.get('generic', 'workspace')).resolve()
#     dict_lock = ws / aiaccel.dict_lock
#     save_result(ws, dict_lock, args.index, result, start_time, end_time)


# if __name__ == '__main__':
#     main()
