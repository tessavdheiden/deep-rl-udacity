from setuptools import setup


setup(
    name='continuous',
    version='0.0.1',
    packages=[
        'models',
        'training',
    ],
    install_requires=[
        'matplotlib',
        'numpy',
        'torch',
        'tensorflow',
    ],
)