from setuptools import find_packages
from setuptools import setup

with open("README.md") as f:
    LONG_DESCRIPTION = f.read()

MAJOR_VERSION = '0'
MINOR_VERSION = '2'
MICRO_VERSION = '21'
VERSION = "{}.{}.{}".format(MAJOR_VERSION, MINOR_VERSION, MICRO_VERSION)

setup(
    name='shrynk',
    version=VERSION,
    description="Using Machine Learning to learn how to Compress",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author='Pascal van Kooten',
    url='https://github.com/kootenpv/shrynk',
    author_email='kootenpv@gmail.com',
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        # '': ['*.txt', '*.rst'],
        'data': ['shrynk/*.gzip']
    },
    install_requires=[
        'pandas',
        'pyarrow==0.12.1',
        'fastparquet',
        "sklearn",
        "preconvert",
        "preconvert_numpy",
        'dill',
        'wrapt_timeout_decorator',
        "just>=0.7.98",
    ],
    extras_require={
        "all": ["fastparquet[brotli,lz4,lzo,snappy,zstandard]"],
        "noc": ["fastparquet[brotli,lz4,snappy]"],
    },
    entry_points={'console_scripts': ['shrynk = shrynk.__main__:main']},
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Customer Service',
        'Intended Audience :: System Administrators',
        'Operating System :: Microsoft',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: System :: Software Distribution',
        'Topic :: System :: Systems Administration',
        'Topic :: Utilities',
    ],
    license='MIT',
    packages=find_packages(),
    zip_safe=False,
    platforms='any',
)
