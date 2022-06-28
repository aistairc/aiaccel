import subprocess
from aiaccel.util import aiaccel


def execute(commands):
    output = subprocess.run(
        commands,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    if output.stderr is not None:
        stderr = output.stderr.decode('UTF-8')
        # エラー処理はしていない。試作中のため。
    if output.stdout is not None:
        stdout = output.stdout.decode('UTF-8')
        return float(stdout.split('\n')[-1])
    print('Error: user.py: execute()')
    return None


def func(p):
    x1 = p["x1"]
    cmd = './mnist_convnet.sh %f' % x1
    y = execute(cmd.split(' '))
    return y


if __name__ == "__main__":

    run = aiaccel.Run()
    run.execute_and_report(func)
