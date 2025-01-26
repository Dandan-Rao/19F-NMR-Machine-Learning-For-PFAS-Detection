# +
random_state = 1

import pandas as pd
from xgboost import XGBRegressor
from rdkit import Chem
import pickle

import common
from hosegen.geometry import *


class getAtomicDescriptorsFrom2DNeighbors:
    # Define a function to get a list of atom indexs in the next spheres
    def findNeighborsNextSphere(
        self, last_sphere_atoms, current_sphere_atoms, next_sphere_level
    ):
        """Get a list of atom indexs in the next spheres

        Parameters
        ----------
        last_sphere_atoms: list
            A list of atoms in the last sphere.

        current_sphere_atoms: list
            A list of atoms in the current sphere

        next_sphere_level: int
             The number of next sphere. For target F atom, the C atom next to it belongs to sphere 0.

        Output
        ----------
        A list of indexs of atoms in the next sphere. If the sphere is No.4. Then the output list will be a list of length 12.
        """
        neighbor_list = []

        # Convert last_sphere_atoms to a set of indices for faster lookups
        if last_sphere_atoms is not None:
            last_sphere_atoms_set = set(
                atom.GetIdx() for atom in last_sphere_atoms if atom is not None
            )
        else:
            last_sphere_atoms_set = set()

        for atom in current_sphere_atoms:
            if atom is None:
                neighbors = [
                    None
                ] * 3  # Adjusted to 3 neighbors. In our case, each atom have at most 4 neighbor atoms.
            else:
                neighbors = (
                    atom.GetNeighbors()
                )  # This function will get all neighbor atoms
                # Filter out atoms that are in last_sphere_atoms
                neighbors = [
                    a for a in neighbors if a.GetIdx() not in last_sphere_atoms_set
                ]

                # Ensure the number of neighbor atoms in the next sphere for each atom is exactly 3.
                if len(neighbors) < 3:
                    neighbors.extend([None] * (3 - len(neighbors)))
                elif len(neighbors) > 3:
                    print(f"More than 3 neighbors: {len(neighbors)}. Trimming to 3.")
                    neighbors = neighbors[:3]

            neighbor_list.extend(neighbors)

        # Final length should be 3^sphere_level
        final_length = 3**next_sphere_level
        if len(neighbor_list) < final_length:
            neighbor_list.extend([None] * (final_length - len(neighbor_list)))

        return neighbor_list

    # Get neighbors of each F atom
    def getNeighborsOfFAtom(self, F_atom):
        """For the target F atom, get neighbor atoms in the 6 neighbor spheres
        Parameters
        ----------
        F_atom: atom object

        Output
        ----------
        sphere0[:1], sphere1, sphere2, sphere3, sphere4, sphere5: list, list, list, list, list, list
            index of atoms in sphere 0, index of atoms in sphere 1, index of atoms in sphere 2, index of atoms in sphere 3,
            index of atoms in sphere 4, and index of atoms in sphere 5
        """
        sphere0_atoms = self.findNeighborsNextSphere(None, [F_atom], 0)
        sphere1_atoms = self.findNeighborsNextSphere(
            [F_atom], sphere0_atoms[:1], 1
        )  # We know sphere0 can only have one valid atom, C
        sphere2_atoms = self.findNeighborsNextSphere(
            sphere0_atoms[:1], sphere1_atoms, 2
        )
        sphere3_atoms = self.findNeighborsNextSphere(sphere1_atoms, sphere2_atoms, 3)
        sphere4_atoms = self.findNeighborsNextSphere(sphere2_atoms, sphere3_atoms, 4)
        sphere5_atoms = self.findNeighborsNextSphere(sphere3_atoms, sphere4_atoms, 5)

        sphere0 = [
            atom.GetIdx() if atom is not None else None for atom in sphere0_atoms
        ]
        sphere1 = [
            atom.GetIdx() if atom is not None else None for atom in sphere1_atoms
        ]
        sphere2 = [
            atom.GetIdx() if atom is not None else None for atom in sphere2_atoms
        ]
        sphere3 = [
            atom.GetIdx() if atom is not None else None for atom in sphere3_atoms
        ]
        sphere4 = [
            atom.GetIdx() if atom is not None else None for atom in sphere4_atoms
        ]
        sphere5 = [
            atom.GetIdx() if atom is not None else None for atom in sphere5_atoms
        ]

        return (
            sphere0[:1],
            sphere1,
            sphere2,
            sphere3,
            sphere4,
            sphere5,
        )  # Only keep the only one valid atom, C

    def getNeighborsInDiffSpheres(self, smiles):
        """
        Parameter
        ----------
        smiles: string
            SMILES of the target compound

        Output
        ----------
        df： Dataframe
            Each line in the df shows the index of neighbor atoms for one F atom in the molecule.
            column name of the df are the number of spheres: 0, 1, 2, 3, 4, 5.
            index of the df are index of F atoms
        """
        # Create an RDKit molecule object from the SMILES string
        mol = Chem.MolFromSmiles(smiles)
        neighbors = {}
        for atom in mol.GetAtoms():
            atom_symbol = (
                atom.GetSymbol()
            )  # Get the atomic symbol (e.g., 'C', 'O', 'F')
            if atom_symbol == "F":
                atom_index = atom.GetIdx()  # Get the index of the atom
                sphere0, sphere1, sphere2, sphere3, sphere4, sphere5 = (
                    self.getNeighborsOfFAtom(atom)
                )
                neighbors[atom_index] = [
                    sphere0,
                    sphere1,
                    sphere2,
                    sphere3,
                    sphere4,
                    sphere5,
                ]

        df = pd.DataFrame.from_dict(neighbors).T
        df = df.map(lambda x: tuple(x) if isinstance(x, list) else x)
        df = df.drop_duplicates()
        return df

    def getFeaturesTableForAtoms(
        self,
        smiles,
        feature_list=[
            "mass",
            "hybridization",
            "isAromatic",
            "degree",
            "valence",
            "explicit_valence",
            "isInRing",
        ],
    ):
        """Select some atomic features, then generate a table showing these features of each atom in the target molecule.
        Parameters
        ----------
        smiles: string
            SMILES of the target molecule
        feature_list: list[string]
            Controls which features will be remained

        Output
        ----------
        df: DataFrame
            columns are atomic features
            index are atom index in the molecule
        """
        mol = Chem.MolFromSmiles(smiles)
        atom_prop = {}
        for atom in mol.GetAtoms():
            atom_symbol = atom.GetSymbol()
            atom_index = atom.GetIdx()
            #             atom_atomicNum = atom.GetAtomicNum()
            atom_mass = atom.GetMass()
            atom_hybridization = atom.GetHybridization()  # SP1, SP2, SP3
            atom_isAromatic = atom.GetIsAromatic()  # bool
            atom_degree = (
                atom.GetDegree()
            )  # The degree of an atom is defined to be its number of directly-bonded neighbors.
            atom_valence = (
                atom.GetTotalValence()
            )  # → int. Returns the total valence (explicit + implicit) of the atom.
            atom_explicit_valence = atom.GetExplicitValence()
            atom_isInRing = atom.IsInRing()
            atom_prop[atom_index] = [
                atom_symbol,
                atom_mass,
                atom_hybridization,
                atom_isAromatic,
                atom_degree,
                atom_valence,
                atom_explicit_valence,
                atom_isInRing,
            ]
            # transform to a dataframe
            df = pd.DataFrame.from_dict(atom_prop).T
            df.columns = [
                "symbol",
                "mass",
                "hybridization",
                "isAromatic",
                "degree",
                "valence",
                "explicit_valence",
                "isInRing",
            ]
            # Convert the column to strings
            df["hybridization"] = df["hybridization"].astype(str).str.slice(-1)
            # # Convert the last character to integers, using errors='coerce' to handle invalid conversion
            df["hybridization"] = pd.to_numeric(df["hybridization"], errors="coerce")
            df[["isAromatic", "isInRing"]] = df[["isAromatic", "isInRing"]].astype(int)
            df = df[feature_list]
        return df

    def removeNoneFromTuple(self, tup):
        # Filter out None values from the tuple and return a new tuple
        return [x for x in tup if x is not None]

    def slimNeighbors(self, neighbors):
        # Apply the function to each element of the DataFrame
        return neighbors.map(self.removeNoneFromTuple)

    def getNeighborsList(self, slim_neighbors, num_spheres):
        # sphere 0 == 1 atom
        # sphere 1 <= 3 atoms
        # sphere 2, leave 9 spaces
        # sphere 3, leave 10 spaces
        # sphere 4, leave 10 spaces
        # if more atoms appear in sphere 2~5, keep only the first 9 or 10 atoms.
        """Reshape the neighbor atoms dataframe for modeling purpose.
        Parameters
        ----------
        slim_neighbors: DataFrame
            A dataframe with each row shows the list of atom indexs in each neighbor spheres of a specific F atom

        Output
        ----------
        df: DataFrame
            A DataFrame with only 33 columns, each column is a neighbor atom slot
            Index of the DataFrame if the atom indexs of F atoms in the molecule.
        """
        dic = {}
        for i, row in slim_neighbors.iterrows():
            neighbors_list = [None] * (1 + 3 + 9 + 10 + 10)
            neighbors_list[0] = row[0][0]
            # Handle sphere 1: up to 3 atoms
            for j in range(min(len(row[1]), 3)):
                neighbors_list[1 + j] = row[1][j]

            # Handle sphere 2: up to 9 atoms
            for j in range(min(len(row[2]), 9)):
                neighbors_list[4 + j] = row[2][j]

            # Handle sphere 3: up to 10 atoms
            for j in range(min(len(row[3]), 10)):
                neighbors_list[13 + j] = row[3][j]

            # Handle sphere 4: up to 10 atoms
            for j in range(min(len(row[4]), 10)):
                neighbors_list[23 + j] = row[4][j]

            # Store the neighbors list in the dictionary
            dic[i] = neighbors_list
            df = pd.DataFrame.from_dict(dic).T
            df.columns = [
                "0_1",
                "1_1",
                "1_2",
                "1_3",
                "2_1",
                "2_2",
                "2_3",
                "2_4",
                "2_5",
                "2_6",
                "2_7",
                "2_8",
                "2_9",
                "3_1",
                "3_2",
                "3_3",
                "3_4",
                "3_5",
                "3_6",
                "3_7",
                "3_8",
                "3_9",
                "3_10",
                "4_1",
                "4_2",
                "4_3",
                "4_4",
                "4_5",
                "4_6",
                "4_7",
                "4_8",
                "4_9",
                "4_10",
            ]
            if num_spheres == 1:
                df = df[["0_1"]]
            if num_spheres == 2:
                df = df[["0_1", "1_1", "1_2", "1_3"]]
            if num_spheres == 3:
                df = df[
                    [
                        "0_1",
                        "1_1",
                        "1_2",
                        "1_3",
                        "2_1",
                        "2_2",
                        "2_3",
                        "2_4",
                        "2_5",
                        "2_6",
                        "2_7",
                        "2_8",
                        "2_9",
                    ]
                ]
            if num_spheres == 4:
                df = df[
                    [
                        "0_1",
                        "1_1",
                        "1_2",
                        "1_3",
                        "2_1",
                        "2_2",
                        "2_3",
                        "2_4",
                        "2_5",
                        "2_6",
                        "2_7",
                        "2_8",
                        "2_9",
                        "3_1",
                        "3_2",
                        "3_3",
                        "3_4",
                        "3_5",
                        "3_6",
                        "3_7",
                        "3_8",
                        "3_9",
                        "3_10",
                    ]
                ]
            if num_spheres == 5:
                df = df[
                    [
                        "0_1",
                        "1_1",
                        "1_2",
                        "1_3",
                        "2_1",
                        "2_2",
                        "2_3",
                        "2_4",
                        "2_5",
                        "2_6",
                        "2_7",
                        "2_8",
                        "2_9",
                        "3_1",
                        "3_2",
                        "3_3",
                        "3_4",
                        "3_5",
                        "3_6",
                        "3_7",
                        "3_8",
                        "3_9",
                        "3_10",
                        "4_1",
                        "4_2",
                        "4_3",
                        "4_4",
                        "4_5",
                        "4_6",
                        "4_7",
                        "4_8",
                        "4_9",
                        "4_10",
                    ]
                ]
        return df

    def getDescriptorsList(self, neighbors_list, atom_features):
        """Combine the list of neighbor atoms with atomic features, create a new df
        Parameters
        ----------
        smiles: string
            SMILES of the target molecule

        Output
        ----------
        df: DataFrame
            A new dataframe contains neighbor atoms and thier atomic features.
        """
        dic = {}
        for i, row in neighbors_list.iterrows():
            descriptors = []
            for neighbor in row:
                if isinstance(neighbor, (int, float)) and not pd.isna(neighbor):
                    neighbor = int(neighbor)
                    descriptors.extend(atom_features.iloc[neighbor, :].values)
                else:
                    # Append a list of Nones if there is no neighbor (at most 6 features as per the example)
                    descriptors.extend([None] * atom_features.shape[1])

            dic[i] = descriptors
        df = pd.DataFrame.from_dict(dic).T
        return df

    def getDescriptorsFromSmiles(
        self,
        smiles,
        num_spheres,
        feature_list=[
            "mass",
            "hybridization",
            "isAromatic",
            "degree",
            "valence",
            "explicit_valence",
            "isInRing",
        ],
    ):
        """Combine all above functions, finish all steps in one function
        Parameters
        ----------
        smiles: string
            SMILES of the target molecule

        Output
        ----------
        df: DataFrame
            A dataframe contains neighbor atoms and thier atomic features.
        """
        feature_table = self.getFeaturesTableForAtoms(smiles, feature_list)
        neighbors_in_diff_spheres = self.getNeighborsInDiffSpheres(smiles)
        slim_neighbors = self.slimNeighbors(neighbors_in_diff_spheres)
        neighbors_list = self.getNeighborsList(slim_neighbors, num_spheres)
        content = self.getDescriptorsList(neighbors_list, feature_table)
        return content

    def getDescriptorsFromDataset(
        self,
        dataset,
        num_spheres,
        feature_list=[
            "mass",
            "hybridization",
            "isAromatic",
            "degree",
            "valence",
            "explicit_valence",
            "isInRing",
        ],
    ):
        """Get a dataframe with each row being the atomic features of neighbor atoms of a target F atom in a fluorinated compound"""
        # Step 1. Transform the column names of the DataFrame to integers where possible and keep them as strings otherwise
        dataset.columns = [common.convert_column_name(name) for name in dataset.columns]
        fluorinated_compounds_content = pd.DataFrame()
        for i, row in dataset.iterrows():
            smiles = row["SMILES"]
            fluorinated_compounds = row["Code"]
            content = self.getDescriptorsFromSmiles(smiles, num_spheres, feature_list)
            index_list = content.index
            try:
                content["NMR_Peaks"] = row[index_list]
            except (KeyError, IndexError):
                pass

            content = content.rename(lambda x: f"{x}_{fluorinated_compounds}")
            fluorinated_compounds_content = pd.concat(
                [fluorinated_compounds_content, content], axis=0
            )
        return fluorinated_compounds_content

    def testXGBoost2DModelPerformance(
        self,
        best_model_file_path,
        dataset,
        num_spheres,
        feature_list=[
            "mass",
            "hybridization",
            "isAromatic",
            "degree",
            "valence",
            "explicit_valence",
            "isInRing",
        ],
    ):
        best_model = XGBRegressor()
        best_model.load_model(best_model_file_path)

        get_2d_descriptors = getAtomicDescriptorsFrom2DNeighbors()
        vali_content = get_2d_descriptors.getDescriptorsFromDataset(
            dataset, num_spheres, feature_list
        )

        vali_content = vali_content.dropna(subset=["NMR_Peaks"])
        y = vali_content["NMR_Peaks"]
        X = vali_content.drop(["NMR_Peaks"], axis=1)

        results_table = common.get_results_table(best_model=best_model, X=X, y=y)
        common.plot_prediction_performance(results_table, figure_title=None)
        common.show_results_scatter(results_table, figure_title=None)
        return results_table


def testRidgePerformance2DFeatures(
    dataset, num_spheres, RidgeCVmodel_path, scaler_path, imputer_path, columns_path
):
    with open(RidgeCVmodel_path, "rb") as file:
        best_model = pickle.load(file)

    with open(scaler_path, "rb") as file:
        scaler = pickle.load(file)

    with open(imputer_path, "rb") as file:
        imputer = pickle.load(file)

    with open(columns_path, "rb") as file:
        train_columns = pickle.load(file)

    get_2d_descriptors = getAtomicDescriptorsFrom2DNeighbors()
    content = get_2d_descriptors.getDescriptorsFromDataset(dataset, num_spheres)

    # Convert all columns in df to numeric where possible, keeping non-numeric values unchanged
    content = content.apply(pd.to_numeric, errors="ignore")

    # Conver column names to 'string'
    content.columns = content.columns.astype(str)

    # Drop rows with NaN values in the 'NMR_Peaks' column
    content = content.dropna(subset=["NMR_Peaks"])

    # Delete columns not shown in the train dataset
    content = content[train_columns]
    content_imputed = imputer.transform(content)

    content_imputed = pd.DataFrame(
        content_imputed, columns=content.columns, index=content.index
    )

    y = content_imputed["NMR_Peaks"]
    X = content_imputed.drop(["NMR_Peaks"], axis=1)

    X_scaled = scaler.transform(X)

    X_scaled = pd.DataFrame(X_scaled)
    X_scaled.columns = X.columns
    X_scaled.index = X.index

    results_table = common.get_results_table(best_model=best_model, X=X_scaled, y=y)
    common.plot_prediction_performance(results_table, figure_title=None)
    common.show_results_scatter(results_table, figure_title=None)
    return results_table


all_neighbor_atoms_list = [
    "SMILES_neighbor10",
    "SMILES_neighbor9",
    "SMILES_neighbor8",
    "SMILES_neighbor7",
    "SMILES_neighbor6",
    "SMILES_neighbor5",
    "SMILES_neighbor4",
    "SMILES_neighbor3",
    "SMILES_neighbor2",
]


# Example
def show_atomic_features_of_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    #     mol = Chem.AddHs(mol)
    atom_info = {}
    for atom in mol.GetAtoms():
        atom_symbol = atom.GetSymbol()
        atom_index = atom.GetIdx()
        #         atom_atomicNum = atom.GetAtomicNum()
        atom_mass = atom.GetMass()
        atom_hybridization = atom.GetHybridization()  # SP1, SP2, SP3
        atom_isAromatic = atom.GetIsAromatic()  # bool
        atom_degree = (
            atom.GetDegree()
        )  # The degree of an atom is defined to be its number of directly-bonded neighbors.
        atom_valence = (
            atom.GetTotalValence()
        )  # → int. Returns the total valence (explicit + implicit) of the atom.
        atom_explicit_valence = atom.GetExplicitValence()
        #         atom_formal_charge = atom.GetFormalCharge()
        #         atom_implicit_valence = atom.GetImplicitValence()
        atom_isInRing = atom.IsInRing()
        #         atom_MonomerInfo = atom.GetMonomerInfo()

        atom_info[atom_index] = [
            atom_mass,
            atom_hybridization,
            atom_isAromatic,
            atom_degree,
            atom_valence,
            atom_explicit_valence,
            atom_isInRing,
        ]

    df = pd.DataFrame(atom_info).T
    df.columns = [
        "mass",
        "hybridization",
        "isAromatic",
        "degree",
        "valence",
        "explicit_valence",
        "isInRing",
    ]
    df["hybridization"] = df["hybridization"].astype(str).str.slice(-1)
    # # Convert the last character to integers, using errors='coerce' to handle invalid conversion
    df["hybridization"] = pd.to_numeric(df["hybridization"], errors="coerce")
    df[["isAromatic", "isInRing"]] = df[["isAromatic", "isInRing"]].astype(int)

    return df
