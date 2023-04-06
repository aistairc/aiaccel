from pathlib import Path
from setuptools import setup
import codecs

here = Path(__file__).resolve().parent

with codecs.open(here / 'requirements.txt', 'r') as fh:
    requirements = [line.replace('\n', '') for line in fh.readlines()]

with codecs.open(here / 'extensions.txt', 'r') as fh:
    extensions = [line.replace('\n', '') for line in fh.readlines()]


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
        'aiaccel.master.evaluator',
        'aiaccel.master.verification',
        'aiaccel.optimizer',
        'aiaccel.scheduler',
        'aiaccel.scheduler.algorithm',
        'aiaccel.scheduler.job',
        'aiaccel.scheduler.job.model',
        'aiaccel.util',
        'aiaccel.cli',
        'aiaccel.storage'
    ],
    data_files=[('aiaccel', ['aiaccel/default_config.yaml'])],
    install_requires=requirements,
    zip_safe=False,
    extensions=extensions,
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "pytest-cov"],
    entry_points={
        'console_scripts': [
            'aiaccel-plot=aiaccel.cli.plot:main',
            'aiaccel-report=aiaccel.cli.report:main',
            'aiaccel-start=aiaccel.cli.start:main',
            'aiaccel-view=aiaccel.cli.view:main',
            'aiaccel-set-result=aiaccel.cli.set_result:main',
        ],
    },
)
