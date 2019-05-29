import argparse
import os
import pathlib
import shutil
import subprocess
import sys


# def main():
#     ''' python runner.py -rn train1 -c "python --val main.py" '''
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--run_name', '-rn', default='default', help='name of experiment')
#     parser.add_argument('--force', '-f', action='store_true')
#     parser.add_argument('--command', '-c', required=True)
#     args = parser.parse_args()

#     src_dir = pathlib.Path.cwd()    # current working directory

#     # check run name
#     run_dir = src_dir.parent / 'playground' / args.run_name
#     if run_dir.is_dir():
#         while not args.force:
#             print('run name %s exists, overwrite or not [Y/n] '
#                 % args.run_name, end = '')
#             Yn = input().strip()
#             if Yn in ['Y']:
#                 break
#             elif Yn in ['N', 'n']:
#                 sys.exit()
#             else:
#                 continue
#         shutil.rmtree(str(run_dir))

#     # copy source files
#     run_dir.mkdir(parents=True, exist_ok=False)
#     dst_dir = run_dir / src_dir.name
#     shutil.copytree(str(src_dir), str(dst_dir))

#     # execution
#     os.chdir(str(dst_dir)) # change current directory
#     env = os.environ.copy()
#     env['run_name'] = args.run_name
#     process = subprocess.Popen(args.command, shell=True, env=env) # execute child program main.py
#     # kill this child process when done
#     while True:
#         try:
#             process.wait()
#             break
#         except KeyboardInterrupt:
#             print('\tPlease double press Ctrl-C within 1 second', flush=True)

if __name__ == '__main__':
    main()