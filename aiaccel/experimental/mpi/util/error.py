import inspect
import traceback

from aiaccel.experimental.mpi.util.time import get_now_str


class MpiError(Exception):
    def __init__(self, message, path=None):
        strace = message
        try:
            super().__init__(message)
            stime = get_now_str()
            st = inspect.stack()[1]  # stack
            fr = st[0]  # frame
            fi = inspect.getframeinfo(fr)  # frame info
            strace += f'''
[{fi.lineno}, {fi.function}, {fi.filename}, {stime}]
{traceback.format_exc()}
'''
            if path is not None:
                with path.open('a') as f:
                    f.write(strace)
        finally:
            print(strace)
