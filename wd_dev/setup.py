from pathlib import Path
from setuptools import setup
import codecs

here = Path(__file__).resolve().parent

with codecs.open(here.joinpath('requirements.txt'), 'r') as fh:
    requirements = [line.replace('\n', '') for line in fh.readlines()]
    requirements = [line.split('==')[0] for line in requirements]

setup(
    name='wd',
    version='0.0.1',
    description='support tool for ABCI and AIST hyperparameter optimizer',
    url='https://gitlab.com/onishi-lab/opt/wd_dev',
    license='MIT',
    packages=[
        'wd', 'wd.bin', 'wd.daemon', 'wd.script', 'wd.wrapper'
    ],
    install_requires=requirements,
    zip_safe=False
)
