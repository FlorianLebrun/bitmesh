from setuptools import setup, find_packages

setup(
    name='playground',
    version='0.1.0',
    packages=find_packages(exclude=['tests*']),
    install_requires=[
        'numpy>=1.19.0',
        'matplotlib>=3.3.0',
    ],
    entry_points={
        'console_scripts': [
            # Add any console scripts here, if any
        ],
    },
)
