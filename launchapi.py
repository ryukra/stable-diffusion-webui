
import os
import sys
import shlex
commandline_args = os.environ.get('COMMANDLINE_ARGS', "")
args = shlex.split(commandline_args)
def extract_arg(args, name):
    return [x for x in args if x != name], name in args
args, skip_torch_cuda_test = extract_arg(args, '--skip-torch-cuda-test')
xformers = '--xformers' in args
sys.argv += args

if "--exit" in args:
    print("Exiting because of --exit argument")
    exit(0)

def start_webui():
    print(f"Launching Web UI with arguments: {' '.join(sys.argv[1:])}")
    import api
    api.webui()

if __name__ == "__main__":
    start_webui()
