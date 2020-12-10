from distutils.core import setup

setup(
    name="Normalizing Flow Network",
    version='1.0',
    author='Simon Böhm',
    install_requires=[
        'tensorflow',
        'tensorflow_probability',
        'scikit_learn',
        'matplotlib'
    ]
)
