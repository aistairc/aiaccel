from __future__ import annotations

from sys import argv
from subprocess import run, PIPE, STDOUT, call
from xml.etree import ElementTree as ET


def main() -> None:  # pragma: no cover
    # node_num = int(argv[-2])
    cmd = argv[-1]
    proc = run('qstat -xml'.split(' '), stdout=PIPE, stderr=STDOUT)
    xml = proc.stdout.decode('utf8')
    root = ET.fromstring(xml)
    for a in root.iter('JB_job_number'):
        id = a.text
    for a in root.iter('queue_name'):
        b: str | None = a.text
        if b is not None:
            host = b[4:]
        else:
            raise Exception('error: queue_name is None.')
    call(f"JOB_ID={id} SGE_TASK_ID=undefined qrsh -inherit {host} '{cmd}'", shell=True)


if __name__ == "__main__":  # pragma: no cover
    main()
