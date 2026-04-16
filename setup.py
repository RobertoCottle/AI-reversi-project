from setuptools import setup, find_packages

setup(
    name="expert_q_learning",
    version="0.1.0",
    packages=find_packages(),
    py_modules=['reversi', 'reversi_server'],
)