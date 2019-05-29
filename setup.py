from setuptools import setup

setup(
    name="testing",
    version="0.1",
    packages=['testing'],
    python_requires='>=3.7',
    install_requires=[
        'tabulate',
        'numpy',
        'scipy',
    ],
)
