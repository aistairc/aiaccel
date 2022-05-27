import subprocess
from aiaccel.util import opt


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
    x2 = p["x2"]
    cmd = './mnist_convnet.sh %f %f' % (x1, x2)
    y = execute(cmd.split(' '))
    return y


if __name__ == "__main__":

    run = opt.Run()
    run.execute_and_report(func)
