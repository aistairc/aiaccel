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
        # No error handling. for testing purposes.
    if output.stdout is not None:
        stdout = output.stdout.decode('UTF-8')
        return float(stdout.split('\n')[-1])
    print('Error: user.py: execute()')
    return None


def execute_debug(commands):
    proc = subprocess.Popen(
        commands,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    save_line = ''
    while True:
        if proc.stdout is None:
            break
        line = proc.stdout.readline().decode().strip()
        if line:
            print(line)
            save_line = line
        else:
            if proc.poll() is not None:
                o, e = proc.communicate()
                s = ''
                if o:
                    s += o.decode().strip()
                if s != '':
                    save_line = s
                if e:
                    s += e.decode().strip()
                print(f'before break: s=|{s}| save_line=|{save_line}|')
                break
    ret_s = save_line.split('\n')[-1]
    print(f'end: save_line=|{save_line}| ret_s=|{ret_s}|')
    return float(ret_s)


def main(p):
    x1 = p["x1"]
    cmd = f'./mnist.sh {x1}'
    y = execute_debug(cmd.split(' '))
    return y


if __name__ == "__main__":
    run = aiaccel.Run()
    run.execute_and_report(main)
