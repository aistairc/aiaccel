import xml.etree.ElementTree as ElementTree

from aiaccel.abci.qstat import parse_job_list, parse_qstat


def test_parse_qstat(data_dir, load_test_config):
    config = load_test_config()
    xml_path = data_dir.joinpath('qstat.xml')

    with open(xml_path, 'r') as f:
        xml_string = f.read()

    stat_list = parse_qstat(config, xml_string)
    assert type(stat_list) is list

def test_parse_job_list(data_dir, load_test_config):
    config = load_test_config()
    xml_path = data_dir.joinpath('qstat.xml')

    with open(xml_path, 'r') as f:
        xml_string = f.read()

    root = ElementTree.fromstring(xml_string)

    for i in root.findall('./job_info/job_list'):
        parse_job_list(config, i)

    assert parse_job_list(config, '') == []
