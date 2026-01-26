"""
Setup script for active_learning_oer package.
"""

from setuptools import setup, find_packages
import os

# Read README
readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
if os.path.exists(readme_path):
    with open(readme_path, 'r', encoding='utf-8') as fh:
        long_description = fh.read()
else:
    long_description = 'Active learning framework for OER catalyst discovery'

# Read requirements
requirements = [
    'numpy>=1.21.2',
    'scipy>=1.7.1',
    'pandas>=1.3.3',
    'matplotlib>=3.4.3',
    'scikit-learn>=0.24.2',
    'ase',
]

setup(
    name='active_learning_oer',
    version='1.0.0',
    author='Minhee Park, Hayun Jeon',
    author_email='minhee2043@snu.ac.kr',
    description='Active learning framework for OER catalyst discovery using Gaussian Process Regression',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/minhee2043/ActiveLearning_OER',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'dev': ['pytest>=6.0', 'black', 'flake8', 'jupyter'],
        'docs': ['sphinx', 'sphinx-rtd-theme'],
    },
    include_package_data=True,
    zip_safe=False,
)
