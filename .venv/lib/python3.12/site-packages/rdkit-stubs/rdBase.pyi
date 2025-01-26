"""
Module containing basic definitions for wrapped C++ code

"""
from __future__ import annotations
import typing
__all__ = ['AttachFileToLog', 'BlockLogs', 'DisableLog', 'EnableLog', 'LogDebugMsg', 'LogErrorMsg', 'LogInfoMsg', 'LogMessage', 'LogStatus', 'LogToCppStreams', 'LogToPythonLogger', 'LogToPythonStderr', 'LogWarningMsg', 'MatchTypeVect', 'SeedRandomNumberGenerator', 'UnsignedLong_Vect', 'VectSizeT', 'VectorOfStringVectors', 'WrapLogs', 'boostVersion', 'ostream', 'rdkitBuild', 'rdkitVersion', 'std_ostream', 'streambuf']
class BlockLogs(Boost.Python.instance):
    """
    Temporarily block logs from outputting while this instance is in scope.
    """
    __instance_size__: typing.ClassVar[int] = 32
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    def __enter__(self) -> BlockLogs:
        """
            C++ signature :
                BlockLogs* __enter__(BlockLogs {lvalue})
        """
    def __exit__(self, exc_type: typing.Any, exc_value: typing.Any, traceback: typing.Any) -> None:
        """
            C++ signature :
                void __exit__(BlockLogs {lvalue},boost::python::api::object,boost::python::api::object,boost::python::api::object)
        """
    def __init__(self) -> None:
        """
            C++ signature :
                void __init__(_object*)
        """
class MatchTypeVect(Boost.Python.instance):
    __instance_size__: typing.ClassVar[int] = 48
    @staticmethod
    def __iter__(vect):
        ...
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    def __contains__(self, item: typing.Any) -> bool:
        """
            C++ signature :
                bool __contains__(std::__1::vector<std::__1::pair<int, int>, std::__1::allocator<std::__1::pair<int, int>>> {lvalue},_object*)
        """
    def __delitem__(self, item: typing.Any) -> None:
        """
            C++ signature :
                void __delitem__(std::__1::vector<std::__1::pair<int, int>, std::__1::allocator<std::__1::pair<int, int>>> {lvalue},_object*)
        """
    def __getitem__(self, item: typing.Any) -> typing.Any:
        """
            C++ signature :
                boost::python::api::object __getitem__(boost::python::back_reference<std::__1::vector<std::__1::pair<int, int>, std::__1::allocator<std::__1::pair<int, int>>>&>,_object*)
        """
    def __init__(self) -> None:
        """
            C++ signature :
                void __init__(_object*)
        """
    def __len__(self) -> int:
        """
            C++ signature :
                unsigned long __len__(std::__1::vector<std::__1::pair<int, int>, std::__1::allocator<std::__1::pair<int, int>>> {lvalue})
        """
    def __setitem__(self, item: typing.Any, value: typing.Any) -> None:
        """
            C++ signature :
                void __setitem__(std::__1::vector<std::__1::pair<int, int>, std::__1::allocator<std::__1::pair<int, int>>> {lvalue},_object*,_object*)
        """
    def append(self, item: typing.Any) -> None:
        """
            C++ signature :
                void append(std::__1::vector<std::__1::pair<int, int>, std::__1::allocator<std::__1::pair<int, int>>> {lvalue},boost::python::api::object)
        """
    def extend(self, other: typing.Any) -> None:
        """
            C++ signature :
                void extend(std::__1::vector<std::__1::pair<int, int>, std::__1::allocator<std::__1::pair<int, int>>> {lvalue},boost::python::api::object)
        """
class UnsignedLong_Vect(Boost.Python.instance):
    __instance_size__: typing.ClassVar[int] = 48
    @staticmethod
    def __iter__(vect):
        ...
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    def __contains__(self, item: typing.Any) -> bool:
        """
            C++ signature :
                bool __contains__(std::__1::vector<unsigned long, std::__1::allocator<unsigned long>> {lvalue},_object*)
        """
    def __delitem__(self, item: typing.Any) -> None:
        """
            C++ signature :
                void __delitem__(std::__1::vector<unsigned long, std::__1::allocator<unsigned long>> {lvalue},_object*)
        """
    def __getitem__(self, item: typing.Any) -> typing.Any:
        """
            C++ signature :
                boost::python::api::object __getitem__(boost::python::back_reference<std::__1::vector<unsigned long, std::__1::allocator<unsigned long>>&>,_object*)
        """
    def __init__(self) -> None:
        """
            C++ signature :
                void __init__(_object*)
        """
    def __len__(self) -> int:
        """
            C++ signature :
                unsigned long __len__(std::__1::vector<unsigned long, std::__1::allocator<unsigned long>> {lvalue})
        """
    def __setitem__(self, item: typing.Any, value: typing.Any) -> None:
        """
            C++ signature :
                void __setitem__(std::__1::vector<unsigned long, std::__1::allocator<unsigned long>> {lvalue},_object*,_object*)
        """
    def append(self, item: typing.Any) -> None:
        """
            C++ signature :
                void append(std::__1::vector<unsigned long, std::__1::allocator<unsigned long>> {lvalue},boost::python::api::object)
        """
    def extend(self, other: typing.Any) -> None:
        """
            C++ signature :
                void extend(std::__1::vector<unsigned long, std::__1::allocator<unsigned long>> {lvalue},boost::python::api::object)
        """
class VectSizeT(Boost.Python.instance):
    __instance_size__: typing.ClassVar[int] = 48
    @staticmethod
    def __iter__(vect):
        ...
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    def __contains__(self, item: typing.Any) -> bool:
        """
            C++ signature :
                bool __contains__(std::__1::vector<unsigned long long, std::__1::allocator<unsigned long long>> {lvalue},_object*)
        """
    def __delitem__(self, item: typing.Any) -> None:
        """
            C++ signature :
                void __delitem__(std::__1::vector<unsigned long long, std::__1::allocator<unsigned long long>> {lvalue},_object*)
        """
    def __getitem__(self, item: typing.Any) -> typing.Any:
        """
            C++ signature :
                boost::python::api::object __getitem__(boost::python::back_reference<std::__1::vector<unsigned long long, std::__1::allocator<unsigned long long>>&>,_object*)
        """
    def __init__(self) -> None:
        """
            C++ signature :
                void __init__(_object*)
        """
    def __len__(self) -> int:
        """
            C++ signature :
                unsigned long __len__(std::__1::vector<unsigned long long, std::__1::allocator<unsigned long long>> {lvalue})
        """
    def __setitem__(self, item: typing.Any, value: typing.Any) -> None:
        """
            C++ signature :
                void __setitem__(std::__1::vector<unsigned long long, std::__1::allocator<unsigned long long>> {lvalue},_object*,_object*)
        """
    def append(self, item: typing.Any) -> None:
        """
            C++ signature :
                void append(std::__1::vector<unsigned long long, std::__1::allocator<unsigned long long>> {lvalue},boost::python::api::object)
        """
    def extend(self, other: typing.Any) -> None:
        """
            C++ signature :
                void extend(std::__1::vector<unsigned long long, std::__1::allocator<unsigned long long>> {lvalue},boost::python::api::object)
        """
class VectorOfStringVectors(Boost.Python.instance):
    __instance_size__: typing.ClassVar[int] = 48
    @staticmethod
    def __iter__(vect):
        ...
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    def __contains__(self, item: typing.Any) -> bool:
        """
            C++ signature :
                bool __contains__(std::__1::vector<std::__1::vector<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>, std::__1::allocator<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>>>, std::__1::allocator<std::__1::vector<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>, std::__1::allocator<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>>>>> {lvalue},_object*)
        """
    def __delitem__(self, item: typing.Any) -> None:
        """
            C++ signature :
                void __delitem__(std::__1::vector<std::__1::vector<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>, std::__1::allocator<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>>>, std::__1::allocator<std::__1::vector<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>, std::__1::allocator<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>>>>> {lvalue},_object*)
        """
    def __getitem__(self, item: typing.Any) -> typing.Any:
        """
            C++ signature :
                boost::python::api::object __getitem__(boost::python::back_reference<std::__1::vector<std::__1::vector<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>, std::__1::allocator<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>>>, std::__1::allocator<std::__1::vector<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>, std::__1::allocator<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>>>>>&>,_object*)
        """
    def __init__(self) -> None:
        """
            C++ signature :
                void __init__(_object*)
        """
    def __len__(self) -> int:
        """
            C++ signature :
                unsigned long __len__(std::__1::vector<std::__1::vector<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>, std::__1::allocator<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>>>, std::__1::allocator<std::__1::vector<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>, std::__1::allocator<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>>>>> {lvalue})
        """
    def __setitem__(self, item: typing.Any, value: typing.Any) -> None:
        """
            C++ signature :
                void __setitem__(std::__1::vector<std::__1::vector<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>, std::__1::allocator<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>>>, std::__1::allocator<std::__1::vector<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>, std::__1::allocator<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>>>>> {lvalue},_object*,_object*)
        """
    def append(self, item: typing.Any) -> None:
        """
            C++ signature :
                void append(std::__1::vector<std::__1::vector<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>, std::__1::allocator<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>>>, std::__1::allocator<std::__1::vector<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>, std::__1::allocator<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>>>>> {lvalue},boost::python::api::object)
        """
    def extend(self, other: typing.Any) -> None:
        """
            C++ signature :
                void extend(std::__1::vector<std::__1::vector<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>, std::__1::allocator<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>>>, std::__1::allocator<std::__1::vector<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>, std::__1::allocator<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>>>>> {lvalue},boost::python::api::object)
        """
class _listNSt3__16vectorIiNS_9allocatorIiEEEE(Boost.Python.instance):
    __instance_size__: typing.ClassVar[int] = 48
    @staticmethod
    def __iter__(vect):
        ...
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    def __contains__(self, item: typing.Any) -> bool:
        """
            C++ signature :
                bool __contains__(std::__1::list<std::__1::vector<int, std::__1::allocator<int>>, std::__1::allocator<std::__1::vector<int, std::__1::allocator<int>>>> {lvalue},_object*)
        """
    def __delitem__(self, item: typing.Any) -> None:
        """
            C++ signature :
                void __delitem__(std::__1::list<std::__1::vector<int, std::__1::allocator<int>>, std::__1::allocator<std::__1::vector<int, std::__1::allocator<int>>>> {lvalue},_object*)
        """
    def __getitem__(self, item: typing.Any) -> typing.Any:
        """
            C++ signature :
                boost::python::api::object __getitem__(boost::python::back_reference<std::__1::list<std::__1::vector<int, std::__1::allocator<int>>, std::__1::allocator<std::__1::vector<int, std::__1::allocator<int>>>>&>,_object*)
        """
    def __init__(self) -> None:
        """
            C++ signature :
                void __init__(_object*)
        """
    def __len__(self) -> int:
        """
            C++ signature :
                unsigned long __len__(std::__1::list<std::__1::vector<int, std::__1::allocator<int>>, std::__1::allocator<std::__1::vector<int, std::__1::allocator<int>>>> {lvalue})
        """
    def __setitem__(self, item: typing.Any, value: typing.Any) -> None:
        """
            C++ signature :
                void __setitem__(std::__1::list<std::__1::vector<int, std::__1::allocator<int>>, std::__1::allocator<std::__1::vector<int, std::__1::allocator<int>>>> {lvalue},_object*,_object*)
        """
class _listNSt3__16vectorIjNS_9allocatorIjEEEE(Boost.Python.instance):
    __instance_size__: typing.ClassVar[int] = 48
    @staticmethod
    def __iter__(vect):
        ...
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    def __contains__(self, item: typing.Any) -> bool:
        """
            C++ signature :
                bool __contains__(std::__1::list<std::__1::vector<unsigned int, std::__1::allocator<unsigned int>>, std::__1::allocator<std::__1::vector<unsigned int, std::__1::allocator<unsigned int>>>> {lvalue},_object*)
        """
    def __delitem__(self, item: typing.Any) -> None:
        """
            C++ signature :
                void __delitem__(std::__1::list<std::__1::vector<unsigned int, std::__1::allocator<unsigned int>>, std::__1::allocator<std::__1::vector<unsigned int, std::__1::allocator<unsigned int>>>> {lvalue},_object*)
        """
    def __getitem__(self, item: typing.Any) -> typing.Any:
        """
            C++ signature :
                boost::python::api::object __getitem__(boost::python::back_reference<std::__1::list<std::__1::vector<unsigned int, std::__1::allocator<unsigned int>>, std::__1::allocator<std::__1::vector<unsigned int, std::__1::allocator<unsigned int>>>>&>,_object*)
        """
    def __init__(self) -> None:
        """
            C++ signature :
                void __init__(_object*)
        """
    def __len__(self) -> int:
        """
            C++ signature :
                unsigned long __len__(std::__1::list<std::__1::vector<unsigned int, std::__1::allocator<unsigned int>>, std::__1::allocator<std::__1::vector<unsigned int, std::__1::allocator<unsigned int>>>> {lvalue})
        """
    def __setitem__(self, item: typing.Any, value: typing.Any) -> None:
        """
            C++ signature :
                void __setitem__(std::__1::list<std::__1::vector<unsigned int, std::__1::allocator<unsigned int>>, std::__1::allocator<std::__1::vector<unsigned int, std::__1::allocator<unsigned int>>>> {lvalue},_object*,_object*)
        """
class _listi(Boost.Python.instance):
    __instance_size__: typing.ClassVar[int] = 48
    @staticmethod
    def __iter__(vect):
        ...
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    def __contains__(self, item: typing.Any) -> bool:
        """
            C++ signature :
                bool __contains__(std::__1::list<int, std::__1::allocator<int>> {lvalue},_object*)
        """
    def __delitem__(self, item: typing.Any) -> None:
        """
            C++ signature :
                void __delitem__(std::__1::list<int, std::__1::allocator<int>> {lvalue},_object*)
        """
    def __getitem__(self, item: typing.Any) -> typing.Any:
        """
            C++ signature :
                boost::python::api::object __getitem__(boost::python::back_reference<std::__1::list<int, std::__1::allocator<int>>&>,_object*)
        """
    def __init__(self) -> None:
        """
            C++ signature :
                void __init__(_object*)
        """
    def __len__(self) -> int:
        """
            C++ signature :
                unsigned long __len__(std::__1::list<int, std::__1::allocator<int>> {lvalue})
        """
    def __setitem__(self, item: typing.Any, value: typing.Any) -> None:
        """
            C++ signature :
                void __setitem__(std::__1::list<int, std::__1::allocator<int>> {lvalue},_object*,_object*)
        """
class _vectNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEE(Boost.Python.instance):
    __instance_size__: typing.ClassVar[int] = 48
    @staticmethod
    def __iter__(vect):
        ...
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    def __contains__(self, item: typing.Any) -> bool:
        """
            C++ signature :
                bool __contains__(std::__1::vector<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>, std::__1::allocator<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>>> {lvalue},_object*)
        """
    def __delitem__(self, item: typing.Any) -> None:
        """
            C++ signature :
                void __delitem__(std::__1::vector<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>, std::__1::allocator<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>>> {lvalue},_object*)
        """
    def __getitem__(self, item: typing.Any) -> typing.Any:
        """
            C++ signature :
                boost::python::api::object __getitem__(boost::python::back_reference<std::__1::vector<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>, std::__1::allocator<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>>>&>,_object*)
        """
    def __init__(self) -> None:
        """
            C++ signature :
                void __init__(_object*)
        """
    def __len__(self) -> int:
        """
            C++ signature :
                unsigned long __len__(std::__1::vector<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>, std::__1::allocator<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>>> {lvalue})
        """
    def __setitem__(self, item: typing.Any, value: typing.Any) -> None:
        """
            C++ signature :
                void __setitem__(std::__1::vector<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>, std::__1::allocator<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>>> {lvalue},_object*,_object*)
        """
    def append(self, item: typing.Any) -> None:
        """
            C++ signature :
                void append(std::__1::vector<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>, std::__1::allocator<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>>> {lvalue},boost::python::api::object)
        """
    def extend(self, other: typing.Any) -> None:
        """
            C++ signature :
                void extend(std::__1::vector<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>, std::__1::allocator<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>>> {lvalue},boost::python::api::object)
        """
class _vectNSt3__16vectorIdNS_9allocatorIdEEEE(Boost.Python.instance):
    __instance_size__: typing.ClassVar[int] = 48
    @staticmethod
    def __iter__(vect):
        ...
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    def __contains__(self, item: typing.Any) -> bool:
        """
            C++ signature :
                bool __contains__(std::__1::vector<std::__1::vector<double, std::__1::allocator<double>>, std::__1::allocator<std::__1::vector<double, std::__1::allocator<double>>>> {lvalue},_object*)
        """
    def __delitem__(self, item: typing.Any) -> None:
        """
            C++ signature :
                void __delitem__(std::__1::vector<std::__1::vector<double, std::__1::allocator<double>>, std::__1::allocator<std::__1::vector<double, std::__1::allocator<double>>>> {lvalue},_object*)
        """
    def __getitem__(self, item: typing.Any) -> typing.Any:
        """
            C++ signature :
                boost::python::api::object __getitem__(boost::python::back_reference<std::__1::vector<std::__1::vector<double, std::__1::allocator<double>>, std::__1::allocator<std::__1::vector<double, std::__1::allocator<double>>>>&>,_object*)
        """
    def __init__(self) -> None:
        """
            C++ signature :
                void __init__(_object*)
        """
    def __len__(self) -> int:
        """
            C++ signature :
                unsigned long __len__(std::__1::vector<std::__1::vector<double, std::__1::allocator<double>>, std::__1::allocator<std::__1::vector<double, std::__1::allocator<double>>>> {lvalue})
        """
    def __setitem__(self, item: typing.Any, value: typing.Any) -> None:
        """
            C++ signature :
                void __setitem__(std::__1::vector<std::__1::vector<double, std::__1::allocator<double>>, std::__1::allocator<std::__1::vector<double, std::__1::allocator<double>>>> {lvalue},_object*,_object*)
        """
    def append(self, item: typing.Any) -> None:
        """
            C++ signature :
                void append(std::__1::vector<std::__1::vector<double, std::__1::allocator<double>>, std::__1::allocator<std::__1::vector<double, std::__1::allocator<double>>>> {lvalue},boost::python::api::object)
        """
    def extend(self, other: typing.Any) -> None:
        """
            C++ signature :
                void extend(std::__1::vector<std::__1::vector<double, std::__1::allocator<double>>, std::__1::allocator<std::__1::vector<double, std::__1::allocator<double>>>> {lvalue},boost::python::api::object)
        """
class _vectNSt3__16vectorIiNS_9allocatorIiEEEE(Boost.Python.instance):
    __instance_size__: typing.ClassVar[int] = 48
    @staticmethod
    def __iter__(vect):
        ...
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    def __contains__(self, item: typing.Any) -> bool:
        """
            C++ signature :
                bool __contains__(std::__1::vector<std::__1::vector<int, std::__1::allocator<int>>, std::__1::allocator<std::__1::vector<int, std::__1::allocator<int>>>> {lvalue},_object*)
        """
    def __delitem__(self, item: typing.Any) -> None:
        """
            C++ signature :
                void __delitem__(std::__1::vector<std::__1::vector<int, std::__1::allocator<int>>, std::__1::allocator<std::__1::vector<int, std::__1::allocator<int>>>> {lvalue},_object*)
        """
    def __getitem__(self, item: typing.Any) -> typing.Any:
        """
            C++ signature :
                boost::python::api::object __getitem__(boost::python::back_reference<std::__1::vector<std::__1::vector<int, std::__1::allocator<int>>, std::__1::allocator<std::__1::vector<int, std::__1::allocator<int>>>>&>,_object*)
        """
    def __init__(self) -> None:
        """
            C++ signature :
                void __init__(_object*)
        """
    def __len__(self) -> int:
        """
            C++ signature :
                unsigned long __len__(std::__1::vector<std::__1::vector<int, std::__1::allocator<int>>, std::__1::allocator<std::__1::vector<int, std::__1::allocator<int>>>> {lvalue})
        """
    def __setitem__(self, item: typing.Any, value: typing.Any) -> None:
        """
            C++ signature :
                void __setitem__(std::__1::vector<std::__1::vector<int, std::__1::allocator<int>>, std::__1::allocator<std::__1::vector<int, std::__1::allocator<int>>>> {lvalue},_object*,_object*)
        """
    def append(self, item: typing.Any) -> None:
        """
            C++ signature :
                void append(std::__1::vector<std::__1::vector<int, std::__1::allocator<int>>, std::__1::allocator<std::__1::vector<int, std::__1::allocator<int>>>> {lvalue},boost::python::api::object)
        """
    def extend(self, other: typing.Any) -> None:
        """
            C++ signature :
                void extend(std::__1::vector<std::__1::vector<int, std::__1::allocator<int>>, std::__1::allocator<std::__1::vector<int, std::__1::allocator<int>>>> {lvalue},boost::python::api::object)
        """
class _vectNSt3__16vectorIjNS_9allocatorIjEEEE(Boost.Python.instance):
    __instance_size__: typing.ClassVar[int] = 48
    @staticmethod
    def __iter__(vect):
        ...
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    def __contains__(self, item: typing.Any) -> bool:
        """
            C++ signature :
                bool __contains__(std::__1::vector<std::__1::vector<unsigned int, std::__1::allocator<unsigned int>>, std::__1::allocator<std::__1::vector<unsigned int, std::__1::allocator<unsigned int>>>> {lvalue},_object*)
        """
    def __delitem__(self, item: typing.Any) -> None:
        """
            C++ signature :
                void __delitem__(std::__1::vector<std::__1::vector<unsigned int, std::__1::allocator<unsigned int>>, std::__1::allocator<std::__1::vector<unsigned int, std::__1::allocator<unsigned int>>>> {lvalue},_object*)
        """
    def __getitem__(self, item: typing.Any) -> typing.Any:
        """
            C++ signature :
                boost::python::api::object __getitem__(boost::python::back_reference<std::__1::vector<std::__1::vector<unsigned int, std::__1::allocator<unsigned int>>, std::__1::allocator<std::__1::vector<unsigned int, std::__1::allocator<unsigned int>>>>&>,_object*)
        """
    def __init__(self) -> None:
        """
            C++ signature :
                void __init__(_object*)
        """
    def __len__(self) -> int:
        """
            C++ signature :
                unsigned long __len__(std::__1::vector<std::__1::vector<unsigned int, std::__1::allocator<unsigned int>>, std::__1::allocator<std::__1::vector<unsigned int, std::__1::allocator<unsigned int>>>> {lvalue})
        """
    def __setitem__(self, item: typing.Any, value: typing.Any) -> None:
        """
            C++ signature :
                void __setitem__(std::__1::vector<std::__1::vector<unsigned int, std::__1::allocator<unsigned int>>, std::__1::allocator<std::__1::vector<unsigned int, std::__1::allocator<unsigned int>>>> {lvalue},_object*,_object*)
        """
    def append(self, item: typing.Any) -> None:
        """
            C++ signature :
                void append(std::__1::vector<std::__1::vector<unsigned int, std::__1::allocator<unsigned int>>, std::__1::allocator<std::__1::vector<unsigned int, std::__1::allocator<unsigned int>>>> {lvalue},boost::python::api::object)
        """
    def extend(self, other: typing.Any) -> None:
        """
            C++ signature :
                void extend(std::__1::vector<std::__1::vector<unsigned int, std::__1::allocator<unsigned int>>, std::__1::allocator<std::__1::vector<unsigned int, std::__1::allocator<unsigned int>>>> {lvalue},boost::python::api::object)
        """
class _vectd(Boost.Python.instance):
    __instance_size__: typing.ClassVar[int] = 48
    @staticmethod
    def __iter__(vect):
        ...
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    def __contains__(self, item: typing.Any) -> bool:
        """
            C++ signature :
                bool __contains__(std::__1::vector<double, std::__1::allocator<double>> {lvalue},_object*)
        """
    def __delitem__(self, item: typing.Any) -> None:
        """
            C++ signature :
                void __delitem__(std::__1::vector<double, std::__1::allocator<double>> {lvalue},_object*)
        """
    def __getitem__(self, item: typing.Any) -> typing.Any:
        """
            C++ signature :
                boost::python::api::object __getitem__(boost::python::back_reference<std::__1::vector<double, std::__1::allocator<double>>&>,_object*)
        """
    def __init__(self) -> None:
        """
            C++ signature :
                void __init__(_object*)
        """
    def __len__(self) -> int:
        """
            C++ signature :
                unsigned long __len__(std::__1::vector<double, std::__1::allocator<double>> {lvalue})
        """
    def __setitem__(self, item: typing.Any, value: typing.Any) -> None:
        """
            C++ signature :
                void __setitem__(std::__1::vector<double, std::__1::allocator<double>> {lvalue},_object*,_object*)
        """
    def append(self, item: typing.Any) -> None:
        """
            C++ signature :
                void append(std::__1::vector<double, std::__1::allocator<double>> {lvalue},boost::python::api::object)
        """
    def extend(self, other: typing.Any) -> None:
        """
            C++ signature :
                void extend(std::__1::vector<double, std::__1::allocator<double>> {lvalue},boost::python::api::object)
        """
class _vecti(Boost.Python.instance):
    __instance_size__: typing.ClassVar[int] = 48
    @staticmethod
    def __iter__(vect):
        ...
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    def __contains__(self, item: typing.Any) -> bool:
        """
            C++ signature :
                bool __contains__(std::__1::vector<int, std::__1::allocator<int>> {lvalue},_object*)
        """
    def __delitem__(self, item: typing.Any) -> None:
        """
            C++ signature :
                void __delitem__(std::__1::vector<int, std::__1::allocator<int>> {lvalue},_object*)
        """
    def __getitem__(self, item: typing.Any) -> typing.Any:
        """
            C++ signature :
                boost::python::api::object __getitem__(boost::python::back_reference<std::__1::vector<int, std::__1::allocator<int>>&>,_object*)
        """
    def __init__(self) -> None:
        """
            C++ signature :
                void __init__(_object*)
        """
    def __len__(self) -> int:
        """
            C++ signature :
                unsigned long __len__(std::__1::vector<int, std::__1::allocator<int>> {lvalue})
        """
    def __setitem__(self, item: typing.Any, value: typing.Any) -> None:
        """
            C++ signature :
                void __setitem__(std::__1::vector<int, std::__1::allocator<int>> {lvalue},_object*,_object*)
        """
    def append(self, item: typing.Any) -> None:
        """
            C++ signature :
                void append(std::__1::vector<int, std::__1::allocator<int>> {lvalue},boost::python::api::object)
        """
    def extend(self, other: typing.Any) -> None:
        """
            C++ signature :
                void extend(std::__1::vector<int, std::__1::allocator<int>> {lvalue},boost::python::api::object)
        """
class _vectj(Boost.Python.instance):
    __instance_size__: typing.ClassVar[int] = 48
    @staticmethod
    def __iter__(vect):
        ...
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    def __contains__(self, item: typing.Any) -> bool:
        """
            C++ signature :
                bool __contains__(std::__1::vector<unsigned int, std::__1::allocator<unsigned int>> {lvalue},_object*)
        """
    def __delitem__(self, item: typing.Any) -> None:
        """
            C++ signature :
                void __delitem__(std::__1::vector<unsigned int, std::__1::allocator<unsigned int>> {lvalue},_object*)
        """
    def __getitem__(self, item: typing.Any) -> typing.Any:
        """
            C++ signature :
                boost::python::api::object __getitem__(boost::python::back_reference<std::__1::vector<unsigned int, std::__1::allocator<unsigned int>>&>,_object*)
        """
    def __init__(self) -> None:
        """
            C++ signature :
                void __init__(_object*)
        """
    def __len__(self) -> int:
        """
            C++ signature :
                unsigned long __len__(std::__1::vector<unsigned int, std::__1::allocator<unsigned int>> {lvalue})
        """
    def __setitem__(self, item: typing.Any, value: typing.Any) -> None:
        """
            C++ signature :
                void __setitem__(std::__1::vector<unsigned int, std::__1::allocator<unsigned int>> {lvalue},_object*,_object*)
        """
    def append(self, item: typing.Any) -> None:
        """
            C++ signature :
                void append(std::__1::vector<unsigned int, std::__1::allocator<unsigned int>> {lvalue},boost::python::api::object)
        """
    def extend(self, other: typing.Any) -> None:
        """
            C++ signature :
                void extend(std::__1::vector<unsigned int, std::__1::allocator<unsigned int>> {lvalue},boost::python::api::object)
        """
class ostream(std_ostream):
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    def __init__(self, python_file_obj: typing.Any, buffer_size: int = 0) -> None:
        """
            C++ signature :
                void __init__(_object*,boost::python::api::object {lvalue} [,unsigned long=0])
        """
class std_ostream(Boost.Python.instance):
    @staticmethod
    def __init__(*args, **kwargs):
        """
        Raises an exception
        This class cannot be instantiated from Python
        """
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
class streambuf(Boost.Python.instance):
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    def __init__(self, python_file_obj: typing.Any, buffer_size: int = 0) -> None:
        """
            documentation
        
            C++ signature :
                void __init__(_object*,boost::python::api::object {lvalue} [,unsigned long=0])
        """
def AttachFileToLog(spec: str, filename: str, delay: int = 100) -> None:
    """
        Causes the log to write to a file
    
        C++ signature :
            void AttachFileToLog(std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>,std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>> [,int=100])
    """
def DisableLog(spec: str) -> None:
    """
        C++ signature :
            void DisableLog(std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>)
    """
def EnableLog(spec: str) -> None:
    """
        C++ signature :
            void EnableLog(std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>)
    """
def LogDebugMsg(msg: str) -> None:
    """
        Log a message to the RDKit debug logs
    
        C++ signature :
            void LogDebugMsg(std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>)
    """
def LogErrorMsg(msg: str) -> None:
    """
        Log a message to the RDKit error logs
    
        C++ signature :
            void LogErrorMsg(std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>)
    """
def LogInfoMsg(msg: str) -> None:
    """
        Log a message to the RDKit info logs
    
        C++ signature :
            void LogInfoMsg(std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>)
    """
def LogMessage(spec: str, msg: str) -> None:
    """
        Log a message to any rdApp.* log
    
        C++ signature :
            void LogMessage(std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>,std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>)
    """
def LogStatus() -> str:
    """
        C++ signature :
            std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>> LogStatus()
    """
def LogToCppStreams() -> None:
    """
        Initialize RDKit logs with C++ streams
    
        C++ signature :
            void LogToCppStreams()
    """
def LogToPythonLogger() -> None:
    """
        Initialize RDKit logs with Python's logging module
    
        C++ signature :
            void LogToPythonLogger()
    """
def LogToPythonStderr() -> None:
    """
        Initialize RDKit logs with Python's stderr stream
    
        C++ signature :
            void LogToPythonStderr()
    """
def LogWarningMsg(msg: str) -> None:
    """
        Log a message to the RDKit warning logs
    
        C++ signature :
            void LogWarningMsg(std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>)
    """
def SeedRandomNumberGenerator(seed: int) -> None:
    """
        Provides a seed to the standard C random number generator
        This does not affect pure Python code, but is relevant to some of the RDKit C++ components.
    
        C++ signature :
            void SeedRandomNumberGenerator(unsigned int)
    """
def WrapLogs() -> None:
    """
        Tee the RDKit logs to Python's stderr stream
    
        C++ signature :
            void WrapLogs()
    """
def _version() -> str:
    """
        Deprecated, use the constant rdkitVersion instead
    
        C++ signature :
            std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>> _version()
    """
_iostreamsEnabled: bool = True
_multithreadedEnabled: bool = True
_serializationEnabled: bool = True
boostVersion: str = '1_85'
rdkitBuild: str = 'Darwin|23.6.0|UNIX|AppleClang|64-bit'
rdkitVersion: str = '2024.09.4'
