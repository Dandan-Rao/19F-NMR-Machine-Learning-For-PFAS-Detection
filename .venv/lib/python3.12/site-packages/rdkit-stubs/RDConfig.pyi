"""
 Configuration for the RDKit Python code

"""
from __future__ import annotations
import os as os
import rdkit as rdkit
import sqlite3 as sqlite3
import sys as sys
__all__ = ['ObsoleteCodeError', 'RDCodeDir', 'RDContribDir', 'RDDataDatabase', 'RDDataDir', 'RDDocsDir', 'RDProjDir', 'RDTestDatabase', 'UnimplementedCodeError', 'defaultDBPassword', 'defaultDBUser', 'molViewer', 'os', 'pythonExe', 'pythonTestCommand', 'rdkit', 'rpcTestPort', 'sqlite3', 'sys', 'usePgSQL', 'useSqlLite']
class ObsoleteCodeError(Exception):
    pass
class UnimplementedCodeError(Exception):
    pass
RDCodeDir: str = '/Users/runner/work/rdkit-pypi/rdkit-pypi/build/temp.macosx-11.0-arm64-cpython-312/rdkit_install/lib/python3.12/site-packages/rdkit'
RDContribDir: str = '/Users/runner/work/rdkit-pypi/rdkit-pypi/build/temp.macosx-11.0-arm64-cpython-312/rdkit_install/share/RDKit/Contrib'
RDDataDatabase: str = '/Users/runner/work/rdkit-pypi/rdkit-pypi/build/temp.macosx-11.0-arm64-cpython-312/rdkit_install/share/RDKit/Data/RDData.sqlt'
RDDataDir: str = '/Users/runner/work/rdkit-pypi/rdkit-pypi/build/temp.macosx-11.0-arm64-cpython-312/rdkit_install/share/RDKit/Data'
RDDocsDir: str = '/Users/runner/work/rdkit-pypi/rdkit-pypi/build/temp.macosx-11.0-arm64-cpython-312/rdkit_install/share/RDKit/Docs'
RDProjDir: str = '/Users/runner/work/rdkit-pypi/rdkit-pypi/build/temp.macosx-11.0-arm64-cpython-312/rdkit_install/share/RDKit/Projects'
RDTestDatabase: str = '/Users/runner/work/rdkit-pypi/rdkit-pypi/build/temp.macosx-11.0-arm64-cpython-312/rdkit_install/share/RDKit/Data/RDTests.sqlt'
defaultDBPassword: str = 'masterkey'
defaultDBUser: str = 'sysdba'
molViewer: str = 'PYMOL'
pythonExe: str = '/private/var/folders/ml/x_tvfgn50_s7p67dm1ypcqqm0000gn/T/cibw-run-uga9jr3h/cp312-macosx_arm64/build/venv/bin/python3.12'
pythonTestCommand: str = 'python'
rpcTestPort: int = 8423
usePgSQL: bool = False
useSqlLite: bool = True
