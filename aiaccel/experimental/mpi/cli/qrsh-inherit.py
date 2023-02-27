import sys
import subprocess
import xml.etree.ElementTree as ET


def main() -> None:  # pragma: no cover
    # node_num = int(sys.argv[-2])
    cmd = sys.argv[-1]
    proc = subprocess.run('qstat -xml'.split(' '), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    xml = proc.stdout.decode('utf8')
    root = ET.fromstring(xml)
    for a in root.iter('JB_job_number'):
        id = a.text
    for a in root.iter('queue_name'):
        host = a.text[4:]
    subprocess.call(f"JOB_ID={id} SGE_TASK_ID=undefined qrsh -inherit {host} '{cmd}'", shell=True)


if __name__ == "__main__":  # pragma: no cover
    main()
