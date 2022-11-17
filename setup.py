from pathlib import Path
from setuptools import setup
import codecs

here = Path(__file__).resolve().parent

with codecs.open(here / 'requirements.txt', 'r') as fh:
    requirements = [line.replace('\n', '') for line in fh.readlines()]
    requirements = [line.split('==')[0] for line in requirements]

setup(
    name='aiaccel',
    version='0.0.1',
    description='AIST hyperparameter optimizer',
    url='https://github.com/aistairc/aiaccel',
    license='MIT',
    packages=[
        'aiaccel',
        'aiaccel.abci',
        'aiaccel.master',
        'aiaccel.evaluator',
        'aiaccel.verification',
        'aiaccel.optimizer',
        'aiaccel.scheduler',
        'aiaccel.scheduler.algorithm',
        'aiaccel.scheduler.job',
        'aiaccel.util',
        'aiaccel.cli',
        'aiaccel.storage'
    ],
    install_requires=requirements,
    zip_safe=False,
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "pytest-cov"],
    entry_points={
        'console_scripts': [
            'aiaccel-plot=aiaccel.cli.plot:main',
            'aiaccel-report=aiaccel.cli.report:main',
            'aiaccel-start=aiaccel.cli.start:main',
            'aiaccel-view=aiaccel.cli.view:main',
        ],
    },
)
