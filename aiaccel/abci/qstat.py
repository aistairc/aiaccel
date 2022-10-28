import xml.etree.ElementTree as ElementTree
from typing import List
from xml.etree.ElementTree import Element

from aiaccel.config import Config

''' Example of stat
stat = {
    'job-ID': 12345,
    'prior': 0.25586,
    'name': 'run.sh',
    'user': 'username',
    'state': 'r',
    'submit/start at': '06/27/2018 21:14:49',
    'queue': 'gpu@g0001',
    'jclass': '',
    'slots': 80,
    'ja-task-ID': ''
}
'''


def parse_qstat(config: Config, qstat: str) -> List[dict]:
    """Parse ABCI 'qstat' command result.

    Args:
        config (ConfileWrapper): A configuration object.
        qstat (str): A 'qstat' result.

    Returns:
        List[dict]: A parsed job list from ABCI 'qstat' command.
    """
    root = ElementTree.fromstring(qstat)
    stat_list = []

    for i in root.findall('./queue_info/job_list'):
        stat_list += parse_job_list(config, i)

    for i in root.findall('./job_info/job_list'):
        stat_list += parse_job_list(config, i)

    return stat_list


def parse_job_list(config: Config, job_list: Element) -> List[dict]:
    """Parse from XML element of 'qstat' to a job list.

    Args:
        config (ConfileWrapper): A configuration object.
        job_list (Element): A XML element of 'qstat' command.

    Returns:
        A job list converted from a XML element of 'qstat' command.
    """
    stat_list = []
    job_id = None
    prior = None
    name = None
    user = None
    state = None
    submit_start_at = None
    queue = None
    jclass = None
    slots = None
    ja_task_id = None

    for j in job_list:
        if 'JB_job_number' == j.tag:
            job_id = j.text
        elif 'JAT_prio' == j.tag:
            prior = j.text
        elif 'JB_name' == j.tag:
            name = j.text
        elif 'JB_owner' == j.tag:
            user = j.text
        elif 'state' == j.tag:
            state = j.text
        elif 'JAT_start_time' == j.tag:
            submit_start_at = j.text
        elif 'queue_name' == j.tag:
            queue = j.text
        elif 'jclass_name' == j.tag:
            jclass = j.text
        elif 'slots' == j.tag:
            slots = j.text

    if (job_id is not None and name is not None):
        stat_list.append(
            {
                'job-ID': job_id,
                'prior': prior,
                'name': name,
                'user': user,
                'state': state,
                'submit/start at': submit_start_at,
                'queue': queue,
                'jclass': jclass,
                'slots': slots,
                'ja-task-ID': ja_task_id
            }
        )

    return stat_list
