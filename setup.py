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
    package_dir={'sobol_seq': 'lib/sobol_seq'},
    packages=[
        'aiaccel',
        'aiaccel.abci',
        'aiaccel.evaluator',
        'aiaccel.verification',
        'aiaccel.optimizer',
        'aiaccel.optimizer.grid',
        'aiaccel.optimizer.random',
        'aiaccel.optimizer.nelder_mead',
        'aiaccel.optimizer.sobol',
        'aiaccel.optimizer.tpe',
        'aiaccel.scheduler',
        'aiaccel.scheduler.algorithm',
        'aiaccel.scheduler.job',
        'aiaccel.util',
        'aiaccel.cli',
        'aiaccel.storage',
        'aiaccel.storage.abstruct',
        'aiaccel.storage.alive',
        'aiaccel.storage.error',
        'aiaccel.storage.hp',
        'aiaccel.storage.jobstate',
        'aiaccel.storage.model',
        'aiaccel.storage.output',
        'aiaccel.storage.pid',
        'aiaccel.storage.result',
        'aiaccel.storage.serializer',
        'aiaccel.storage.timestamp',
        'aiaccel.storage.trial',
        'sobol_seq'
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
            'aiaccel-stop=aiaccel.cli.stop:main',
            'aiaccel-view=aiaccel.cli.view:main',
        ],
    },
)
