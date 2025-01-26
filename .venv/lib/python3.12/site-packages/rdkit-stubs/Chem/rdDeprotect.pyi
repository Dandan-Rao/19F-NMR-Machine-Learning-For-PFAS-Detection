from __future__ import annotations
import rdkit.Chem
import typing
__all__ = ['Deprotect', 'DeprotectData', 'DeprotectDataVect', 'DeprotectInPlace', 'GetDeprotections']
class DeprotectData(Boost.Python.instance):
    """
    DeprotectData class, contains a single deprotection reaction and information
    
     deprotectdata.deprotection_class - functional group being protected
     deprotectdata.reaction_smarts - reaction smarts used for deprotection
     deprotectdata.abbreviation - common abbreviation for the protecting group
     deprotectdata.full_name - full name for the protecting group
    
    
    """
    __instance_size__: typing.ClassVar[int] = 160
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    def __init__(self, deprotection_class: str, reaction_smarts: str, abbreviation: str, full_name: str) -> None:
        """
            Construct a new DeprotectData instance.
              >>> reaction_class = "amine"
              >>> reaction_smarts = "[C;R0][C;R0]([C;R0])([O;R0][C;R0](=[O;R0])[NX3;H0,H1:1])C>>[N:1]"
              >>> abbreviation = "Boc"
              >>> full_name = "tert-butyloxycarbonyl"
              >>> data = DeprotectData(reaction_class, reaction_smarts, abbreviation, full_name)
              >>> assert data.isValid()
            
            
        
            C++ signature :
                void __init__(_object*,std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>,std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>,std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>,std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>)
        """
    def isValid(self) -> bool:
        """
            Returns True if the DeprotectData has a valid reaction
        
            C++ signature :
                bool isValid(RDKit::Deprotect::DeprotectData {lvalue})
        """
    @property
    def abbreviation(*args, **kwargs):
        ...
    @property
    def deprotection_class(*args, **kwargs):
        ...
    @property
    def example(*args, **kwargs):
        ...
    @property
    def full_name(*args, **kwargs):
        ...
    @property
    def reaction_smarts(*args, **kwargs):
        ...
class DeprotectDataVect(Boost.Python.instance):
    __instance_size__: typing.ClassVar[int] = 48
    @staticmethod
    def __reduce__(*args, **kwargs):
        ...
    def __contains__(self, item: typing.Any) -> bool:
        """
            C++ signature :
                bool __contains__(std::__1::vector<RDKit::Deprotect::DeprotectData, std::__1::allocator<RDKit::Deprotect::DeprotectData>> {lvalue},_object*)
        """
    def __delitem__(self, item: typing.Any) -> None:
        """
            C++ signature :
                void __delitem__(std::__1::vector<RDKit::Deprotect::DeprotectData, std::__1::allocator<RDKit::Deprotect::DeprotectData>> {lvalue},_object*)
        """
    def __getitem__(self, item: typing.Any) -> typing.Any:
        """
            C++ signature :
                boost::python::api::object __getitem__(boost::python::back_reference<std::__1::vector<RDKit::Deprotect::DeprotectData, std::__1::allocator<RDKit::Deprotect::DeprotectData>>&>,_object*)
        """
    def __init__(self) -> None:
        """
            C++ signature :
                void __init__(_object*)
        """
    def __iter__(self) -> typing.Any:
        """
            C++ signature :
                boost::python::objects::iterator_range<boost::python::return_internal_reference<1ul, boost::python::default_call_policies>, std::__1::__wrap_iter<RDKit::Deprotect::DeprotectData*>> __iter__(boost::python::back_reference<std::__1::vector<RDKit::Deprotect::DeprotectData, std::__1::allocator<RDKit::Deprotect::DeprotectData>>&>)
        """
    def __len__(self) -> int:
        """
            C++ signature :
                unsigned long __len__(std::__1::vector<RDKit::Deprotect::DeprotectData, std::__1::allocator<RDKit::Deprotect::DeprotectData>> {lvalue})
        """
    def __setitem__(self, item: typing.Any, value: typing.Any) -> None:
        """
            C++ signature :
                void __setitem__(std::__1::vector<RDKit::Deprotect::DeprotectData, std::__1::allocator<RDKit::Deprotect::DeprotectData>> {lvalue},_object*,_object*)
        """
    def append(self, item: typing.Any) -> None:
        """
            C++ signature :
                void append(std::__1::vector<RDKit::Deprotect::DeprotectData, std::__1::allocator<RDKit::Deprotect::DeprotectData>> {lvalue},boost::python::api::object)
        """
    def extend(self, other: typing.Any) -> None:
        """
            C++ signature :
                void extend(std::__1::vector<RDKit::Deprotect::DeprotectData, std::__1::allocator<RDKit::Deprotect::DeprotectData>> {lvalue},boost::python::api::object)
        """
def Deprotect(mol: Mol, deprotections: typing.Any = None) -> rdkit.Chem.Mol:
    """
        Return the deprotected version of the molecule.
    
        C++ signature :
            boost::shared_ptr<RDKit::ROMol> Deprotect(RDKit::ROMol [,boost::python::api::object=None])
    """
def DeprotectInPlace(mol: Mol, deprotections: typing.Any = None) -> bool:
    """
        Deprotects the molecule in place.
    
        C++ signature :
            bool DeprotectInPlace(RDKit::ROMol {lvalue} [,boost::python::api::object=None])
    """
def GetDeprotections() -> DeprotectDataVect:
    """
        Return the default list of deprotections
    
        C++ signature :
            std::__1::vector<RDKit::Deprotect::DeprotectData, std::__1::allocator<RDKit::Deprotect::DeprotectData>> GetDeprotections()
    """
