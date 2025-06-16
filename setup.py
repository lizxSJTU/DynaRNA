from setuptools import setup

setup(
    name="DynaRNA",
    packages=[
        'data', 'model', 
    ],
    package_dir={
        'data': './data',
        'model': './model',
    },
)
