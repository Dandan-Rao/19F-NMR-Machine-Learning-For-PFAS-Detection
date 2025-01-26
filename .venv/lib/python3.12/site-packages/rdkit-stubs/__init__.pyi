from __future__ import annotations
import logging as logging
import sys as sys
from .rdBase import *
__all__ = ['VECT_WRAPS', 'VectIter', 'log_handler', 'logger', 'logging', 'name', 'object', 'rdBase', 'sys']
class VectIter:
    def __init__(self, vect):
        ...
    def __iter__(self):
        ...
    def __next__(self):
        ...
def __vect__iter__(vect):
    ...
VECT_WRAPS: set = {'VectorOfStringVectors', 'MatchTypeVect', 'VectSizeT', 'UnsignedLong_Vect'}
__version__: str = '2024.09.4'
log_handler: logging.StreamHandler  # value = <StreamHandler <stderr> (NOTSET)>
logger: logging.Logger  # value = <Logger rdkit (WARNING)>
name: str = '__file__'
object: str = '/Users/runner/work/rdkit-pypi/rdkit-pypi/build/temp.macosx-11.0-arm64-cpython-312/rdkit_install/lib/python3.12/site-packages/rdkit/rdBase.so'
