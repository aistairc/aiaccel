from pathlib import Path
from setuptools import setup
import codecs

here = Path(__file__).resolve().parent

with codecs.open(here / 'requirements.txt', 'r') as fh:
    requirements = [line.replace('\n', '') for line in fh.readlines()]
    requirements = [line.split('==')[0] for line in requirements]

setup(
    name='aiaccel',
    version='0.0.4',
    description='AIST hyperparameter optimizer',
    url='https://gitlab.com/onishi-lab/opt',
    license='MIT',
    package_dir={'sobol_seq': 'lib/sobol_seq'},
    packages=[
        'aiaccel',
        'aiaccel.abci',
        'aiaccel.master',
        'aiaccel.master.evaluator',
        'aiaccel.master.verification',
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
        'sobol_seq'
    ],
    install_requires=requirements,
    zip_safe=False,
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "pytest-cov"]
)
