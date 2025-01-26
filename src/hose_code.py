# +
import pandas as pd
import os

random_state = 1

from rdkit import Chem

import common
from hosegen import HoseGenerator
from hosegen.geometry import *


## Generate HOSE code with different radius
def getHoseCodeContent(data, max_radius=6):
    """For each fluorinated compound in the dataset, retrieve the HOSE code for the F atoms in the molecule.
    Parameters
    ----------
    data: DataFrame
        A dataframe contains SMILES, Code of various fluorinated compounds

    Output
    ----------
    fluorinated_compounds_content: DataFrame
        The index of the DataFrame corresponds to the F atom code, formatted as `'FatomIndex_fluorinated_compoundsCode'`.
        The columns represent radii from 0 to 5.
        For example, the values in column 3 contain the HOSE code that encodes information about neighboring atoms within 3 spheres.
        The final column contains the \(^{19}\)F NMR chemical shift values.
    """
    gen = HoseGenerator()
    # Transform the column names of the DataFrame to integers where possible and keep them as strings otherwise
    data.columns = [common.convert_column_name(name) for name in data.columns]
    fluorinated_compounds_content = pd.DataFrame()

    for i, row in data.iterrows():
        smiles = row["SMILES"]
        fluorinated_compounds = row["Code"]
        mol = Chem.MolFromSmiles(smiles)

        output_file_path = os.path.join(
            "..", "artifacts", "temp", f"{fluorinated_compounds}.mol"
        )
        Chem.MolToMolFile(mol, output_file_path)
        wedgemap = create_wedgemap(output_file_path)
        if os.path.exists(output_file_path):
            os.remove(output_file_path)

        dic = {}
        for atom in mol.GetAtoms():
            atom_symbol = atom.GetSymbol()
            atom_index = atom.GetIdx()
            if atom_symbol == "F":
                hose_codes = []
                for j in range(max_radius):
                    hose_codes.append(
                        gen.get_Hose_codes(
                            mol,
                            atom_index,
                            usestereo=True,
                            max_radius=j + 1,
                            wedgebond=wedgemap,
                        )
                    )
                dic[atom_index] = hose_codes

        df = pd.DataFrame.from_dict(dic).T
        F_indexs = df.index
        df["NMR_Peaks"] = row[F_indexs].values
        df = df.rename(index=lambda x: f"{x}_{fluorinated_compounds}")
        #         df = df.drop_duplicates()
        fluorinated_compounds_content = pd.concat(
            [fluorinated_compounds_content, df], axis=0
        )
    return fluorinated_compounds_content


# +
def getTrainDictionary_HOSE(train_data):
    """
    Get the mean value of 'NMR_Peaks' for each unique values in HOSE code.
    Parameters
    ----------
    train_data: DataFrame
        A datafram with F atom index being df index and radius being column-name. Values are HOSE code of different radius.

    Output
    ----------
    [sphere6_dic, sphere5_dic, sphere4_dic, sphere3_dic, sphere2_dic, sphere1_dic]: A list of dictionaries
    """
    train_data = train_data.dropna(subset=["NMR_Peaks"])
    train_data["NMR_Peaks"] = train_data["NMR_Peaks"].astype(float)

    # Get the mean value of 'NMR_Peaks' for each unique values in column '5'
    grouped_df = train_data[[5, "NMR_Peaks"]].groupby(5)["NMR_Peaks"].mean()
    sphere6_dic = grouped_df.to_dict()

    grouped_df = train_data[[4, "NMR_Peaks"]].groupby(4)["NMR_Peaks"].mean()
    sphere5_dic = grouped_df.to_dict()

    grouped_df = train_data[[3, "NMR_Peaks"]].groupby(3)["NMR_Peaks"].mean()
    sphere4_dic = grouped_df.to_dict()

    grouped_df = train_data[[2, "NMR_Peaks"]].groupby(2)["NMR_Peaks"].mean()
    sphere3_dic = grouped_df.to_dict()

    grouped_df = train_data[[1, "NMR_Peaks"]].groupby(1)["NMR_Peaks"].mean()
    sphere2_dic = grouped_df.to_dict()

    grouped_df = train_data[[0, "NMR_Peaks"]].groupby(0)["NMR_Peaks"].mean()
    sphere1_dic = grouped_df.to_dict()

    return [
        sphere6_dic,
        sphere5_dic,
        sphere4_dic,
        sphere3_dic,
        sphere2_dic,
        sphere1_dic,
    ]


def HOSE_Model(sphere_dics, test_data, mean_value_in_train_data):
    """
    Parameters
    ----------
    sphere_dics: A list of dictionaries
        With max_radius = 6, the list contains 6 dictionary.
        The key of the nth dictionary is the HOSE code with radius being n, and values being the mean 19F NMR
        shift value of the HOSE code in the training dataset.

    test_data: DataFrame
        A dataframe contains the HOSE code for each F atoms in the test dataset.

    mean_value_in_train_data: float
        If no same HOSE code was found in the sphere_dics, use mean_value_in_train_data as prediction value

    Output
    ----------
    prediction: list
        The predicted 19F NMR shift values for F atoms in the test dataset

    similarity_levels: list[int]
        Meaning of the similarity level:
            6: find in pool with max_radius = 6
            5: find in pool with max_radius = 5

            # HOSE code prediction with four or more spheres and respecting stereochemistry are generally considered reliable.
            4: find in pool with max_radius = 4
            3: find in pool with max_radius = 3
            2: find in pool with max_radius = 2
            1: find in pool with max_radius = 1
    """
    #     test_data = test_data.dropna(subset = ['NMR_Peaks'])
    test_data["NMR_Peaks"] = test_data["NMR_Peaks"].apply(
        pd.to_numeric, errors="coerce"
    )

    prediction = []
    similarity_levels = []
    for i, row in test_data.iterrows():
        if row[5] in sphere_dics[0]:
            prediction.append(sphere_dics[0][row[5]])
            similarity_levels.append(6)
        elif row[4] in sphere_dics[1]:
            prediction.append(sphere_dics[1][row[4]])
            similarity_levels.append(5)
        elif row[3] in sphere_dics[2]:
            prediction.append(sphere_dics[2][row[3]])
            similarity_levels.append(4)
        elif row[2] in sphere_dics[3]:
            prediction.append(sphere_dics[3][row[2]])
            similarity_levels.append(3)
        elif row[1] in sphere_dics[4]:
            prediction.append(sphere_dics[4][row[1]])
            similarity_levels.append(2)
        elif row[0] in sphere_dics[5]:
            prediction.append(sphere_dics[5][row[0]])
            similarity_levels.append(1)
        else:
            prediction.append(mean_value_in_train_data)
            similarity_levels.append(0)
    return prediction, similarity_levels


def getResults_HOSE(prediction, similarity_levels, test_data):
    """Reorganize prediction, similarity_levels and combine it with test_data
    Output
    ----------
    results: DataFrame
        Column 'actual': Actual 19F NMR shift values for F atoms in the test dataset
        Column 'prediction': The predicted 19F NMR shift values using the HOSE code method
        Column 'diff': Absolute  error of the prediction
        Column 'similarity_levels': The value Represents how similar a particular F atom is to the F atoms in the training dataset.
        This can be treated as a value between 1 and 6, where 6 represents maximum similarity (i.e., very reliable predictions).
    """
    #     test_data = test_data.dropna(subset = ['NMR_Peaks'])
    test_data["NMR_Peaks"] = test_data["NMR_Peaks"].apply(
        pd.to_numeric, errors="coerce"
    )
    results = test_data[["NMR_Peaks"]].copy()
    results.columns = ["actual"]
    results["prediction"] = prediction
    results["diff"] = results["prediction"] - results["actual"]
    results["diff"] = results["diff"].abs()
    results["similarity_levels"] = similarity_levels
    return results


def get_HOSE_prediction_results_table(
    HOSE_Code_database_file_path, test_fluorinated_compounds
):
    """'
    Parameters
    ----------
    HOSE_Code_database_file_path: str
        Path to the HOSE code database file
    test_fluorinated_compounds: DataFrame
        DataFrame containing the test fluorinated compounds
    Output
    ----------
    results: DataFrame
        DataFrame containing the prediction results
    """
    HOSE_Code_database = pd.read_csv(HOSE_Code_database_file_path)
    # Transform column names to int where possible
    HOSE_Code_database.columns = [
        common.convert_column_name(name) for name in HOSE_Code_database.columns
    ]

    HOSE_codes_test = getHoseCodeContent(test_fluorinated_compounds)

    # Get HOSE Code and corresponding 19F NMR values using train dataset
    sphere_dics = getTrainDictionary_HOSE(HOSE_Code_database)

    HOSE_Code_database["NMR_Peaks"] = HOSE_Code_database["NMR_Peaks"].apply(
        pd.to_numeric, errors="coerce"
    )

    # Get prediction results and corresponding similarity levels for the validation dataset
    prediction, similarity_levels = HOSE_Model(
        sphere_dics, HOSE_codes_test, HOSE_Code_database["NMR_Peaks"].mean()
    )
    # Validation dataset
    results = getResults_HOSE(prediction, similarity_levels, HOSE_codes_test)

    return results
