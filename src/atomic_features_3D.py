import os

random_state = 1

import pandas as pd
from xgboost import XGBRegressor
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import pickle
import subprocess
import common


def get_sdf_file(smiles):
    """
    Convert SMILES to 3D conformation and save as .sdf and .png file.
    The .sdf and .png file will temporarily be stored in the '~/artifacts/temp` directory, with names "temp.sdf" and "temp.png".
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError("Invalid SMILES string")

    # Transform SMILES to canonical SMILES
    smiles = Chem.MolToSmiles(mol)

    mol = Chem.MolFromSmiles(smiles)
    sdf = Transform_SMILE_to_3D_conformation(smiles)

    # Save .sdf file
    file_path = os.path.join("..", "artifacts", "temp", "temp.sdf")

    w = Chem.SDWriter(file_path)
    w.write(sdf)
    w.close()

    # Create pictures for the molecule
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))

        # Save the molecule image with the Code as part of the file name
        file_path = os.path.join("..", "artifacts", "temp", f"temp.png")
        Draw.MolToFile(mol, file_path, size=(600, 400))


def get_descriptors_and_neighbors_info():
    """
    Run Java programs to get CDK descriptors and neighbors information for fluorinated compounds.
    """
    # Define the Java directory
    java_dir = os.path.join("..", "external", "JAVAget_3D_feature_set", "java")

    # Use ".;*" or ".:*" as classpath
    classpath = f".{os.pathsep}cdk-1.4.13.jar:cdk.jar:jmathio.jar:jmathplot.jar"

    try:
        # Run the Java programs
        result = subprocess.run(
            ["java", "-cp", classpath, "GetCDKDescriptors", "temp"],
            check=True,
            text=True,
            capture_output=True,
            cwd=java_dir,  # This is important - sets the working directory
        )

    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        print(e.stderr)
        raise


def get_test_fluorianted_compounds_info(smiles, train_dataset):
    """
    Get the information of fluorinated compounds from the target dataset.
    """
    if smiles in train_dataset["SMILES"].values:
        dataset = train_dataset[train_dataset["SMILES"] == smiles]
        if len(dataset) > 1:
            dataset = dataset.iloc[[0], :]
    #         dataset['Code'] = 'temp'

    else:
        # Define columns from '0' to '70' and add an additional column 'Code'
        columns = ["Code", "SMILES"] + [str(i) for i in range(71)]

        # Create the DataFrame with 71 columns filled with None and 'Code' filled with 'temp'
        dataset = pd.DataFrame([["temp"] + [smiles] + [None] * 71], columns=columns)
    return dataset


def get_features_table(dataset):
    """
    Get the features table containing 3d atomic features for fluorinated compounds.
    """
    neighbor_num = 5
    if dataset["Code"].values == ["temp"]:
        file_path = os.path.join("..", "artifacts", "temp")
        NMR_peaks_with_desc = Combine_descriptors(
            dataset, neighbor_num, file_path, file_path, with_additional_info=False
        )
    else:
        NMR_peaks_with_desc = Combine_descriptors(
            dataset, neighbor_num, with_additional_info=False
        )
    return NMR_peaks_with_desc


def Transform_SMILE_to_3D_conformation(SMILES):
    """
    Convert SMILES to 3D conformation.  Optimize the conformation using MMFF.
    """
    mol = Chem.MolFromSmiles(SMILES)
    hmol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(
        hmol, randomSeed=0xF00D
    )  # attempts to generate a reasonable 3D structure for the molecule by placing the atoms in 3D space, considering the bond lengths, angles, and steric effects.
    AllChem.MMFFOptimizeMolecule(
        hmol
    )  # optimizes the 3D conformation of the molecule using a force field.
    my_embedded_mol = Chem.RemoveHs(hmol)
    return my_embedded_mol


file_path1 = os.path.join("..", "dataset", "descriptors")
file_path2 = os.path.join("..", "dataset", "neighbors")


def Combine_descriptors(
    dataset,
    neighbor_num,
    fluorinated_compounds_CDK_Desc_file_path=file_path1,
    fluorinated_compounds_neighbor_atoms_file_path=file_path2,
    with_additional_info=True,
):
    """
    Parameters
    ----------
    dataset : DataFrame
        Must contain a column 'Code', which will be used to retrieve CDK descriptor files and neighbor atom information from corresponding files.

    neighbor_num : int
        The number of neighboring atoms in 3D space used to extract atomic features.

    fluorinated_compounds_CDK_Desc_file_path : str
        Path to the folder storing CDK descriptor files for all fluorinated compounds. These descriptors are obtained using the CDK library in Java.

    fluorinated_compounds_neighbor_atoms_file_path : str
        Path to the folder storing information about neighboring atoms for all atoms in the fluorinated compounds. This information is obtained using the CDK library in Java.

    with_additional_info: bool
        If True, the 'dataset' DataFrame must also include columns: 'SMILES', 'Compound name', and 'Solvent_used_for_NMR'.

    Returns
    -------
    All_NMR_w_Desc : DataFrame
        A DataFrame containing descriptors from neighboring atoms for each F atom in the fluorinated compounds in the dataset.
    """
    # Transform the column names of the DataFrame to integers where possible and keep them as strings otherwise
    dataset.columns = [common.convert_column_name(name) for name in dataset.columns]

    fluorinated_compounds_collection = []
    All_NMR_w_Desc = pd.DataFrame()

    # The code will be used to match CDK descriptor files and neighbor atoms file.
    for i in dataset["Code"]:
        fluorinated_compounds_collection.append(i)

    for fluorinated_compounds in fluorinated_compounds_collection:
        # Import Descriptors
        fluorinated_compounds_CDK_Desc_path = os.path.join(
            fluorinated_compounds_CDK_Desc_file_path,
            f"{fluorinated_compounds}_Descriptors.csv",
        )
        fluorinated_compounds_CDK_Desc = pd.read_csv(
            fluorinated_compounds_CDK_Desc_path
        )

        # Delete all columns containg 'proton'
        fluorinated_compounds_CDK_Desc = fluorinated_compounds_CDK_Desc.loc[
            :,
            ~fluorinated_compounds_CDK_Desc.columns.str.contains("proton", case=False),
        ]

        # Get index of all F atoms in the fluorinated_compounds molecular
        F_index = fluorinated_compounds_CDK_Desc[
            fluorinated_compounds_CDK_Desc.SMILES == "F"
        ].index

        Desc_of_target_F_atoms = fluorinated_compounds_CDK_Desc[
            fluorinated_compounds_CDK_Desc.SMILES == "F"
        ]

        # Import information about neighbor atoms
        fluorinated_compounds_neighbor_atom_path = os.path.join(
            fluorinated_compounds_neighbor_atoms_file_path,
            f"{fluorinated_compounds}_Neighbors.csv",
        )
        fluorinated_compounds_neighbor_atoms = pd.read_csv(
            fluorinated_compounds_neighbor_atom_path, header=None
        )

        # Generate a df about NMR peak data, which with F index as index and "NMR_Peaks" as col.
        # If the df does not contains NMR peak data, create a blank df
        try:
            NMR_Peaks = dataset[dataset.Code == fluorinated_compounds]
            NMR_Peaks = NMR_Peaks.reset_index(drop=True)
            NMR_Peaks.index = ["NMR_Peaks"]
            NMR_Peaks = NMR_Peaks.loc[:, F_index]
            NMR_Peaks = NMR_Peaks
            NMR_Peaks = NMR_Peaks.T
        except KeyError:
            NMR_Peaks = pd.DataFrame({}, index=F_index)

        # Generate a df about descriptors of target F atoms
        Desc_F_atoms_self = Desc_of_target_F_atoms.rename(columns=lambda x: x + "_self")
        F_NMR_and_Desc = NMR_Peaks.merge(
            Desc_F_atoms_self, left_index=True, right_index=True
        )

        Neighbor_Desc = pd.DataFrame()

        for i in F_index:
            all_neighbors = fluorinated_compounds_neighbor_atoms.shape[1]
            if neighbor_num <= all_neighbors:
                neighbor_index = fluorinated_compounds_neighbor_atoms.iloc[
                    i, range(neighbor_num)
                ]
            else:
                neighbor_index = fluorinated_compounds_neighbor_atoms.iloc[i, :]

            neighbor_desc = fluorinated_compounds_CDK_Desc.iloc[
                neighbor_index
            ].reset_index()
            df = pd.DataFrame()
            for index, row in neighbor_desc.iterrows():
                interm = pd.DataFrame(neighbor_desc.iloc[index]).T.reset_index()
                interm = interm.rename(columns=lambda x: x + f"_neighbor{index+1}")
                df = df.merge(interm, how="outer", left_index=True, right_index=True)
            df.index = [i]
            Neighbor_Desc = pd.concat([Neighbor_Desc, df], axis=0)
        NMR_w_Desc = pd.concat([F_NMR_and_Desc, Neighbor_Desc], axis=1)
        NMR_w_Desc = NMR_w_Desc.rename(lambda x: f"{x}_{fluorinated_compounds}")

        if with_additional_info:
            # Extract single values for the 'SMILES', 'Compound name', and 'Solvent_used_for_NMR' columns
            NMR_w_Desc["SMILES"] = dataset[dataset.Code == fluorinated_compounds][
                "SMILES"
            ].values[0]
            NMR_w_Desc["Compound name"] = dataset[
                dataset.Code == fluorinated_compounds
            ]["Compound name"].values[0]
            NMR_w_Desc["Solvent_used_for_NMR"] = dataset[
                dataset.Code == fluorinated_compounds
            ]["Solvent_used_for_NMR"].values[0]

        All_NMR_w_Desc = pd.concat([All_NMR_w_Desc, NMR_w_Desc], axis=0)

    return All_NMR_w_Desc


def Fatoms_Peak(
    fluorinated_compounds_NMR_Peaks, fluorinated_compounds_CDK_Desc_file_path=file_path1
):
    """
    Parameters
    ----------
    fluorinated_compounds_NMR_Peaks : DataFrame
        A DataFrame containing NMR peak data for fluorinated compounds.
    fluorinated_compounds_CDK_Desc_file_path : str
        Path to the folder storing CDK descriptor files for all fluorinated compounds. These descriptors are obtained using the CDK library in Java.
    Returns
    -------
    all_NMR_Peaks : DataFrame
        A DataFrame list 19F NMR shift for each F atoms.
    """
    fluorinated_compounds_collection = []
    all_NMR_Peaks = pd.DataFrame()
    for i in fluorinated_compounds_NMR_Peaks["Code"]:
        fluorinated_compounds_collection.append(i)

    for fluorinated_compounds in fluorinated_compounds_collection:
        # Import Descriptors
        fluorinated_compounds_CDK_Desc_path = os.path.join(
            fluorinated_compounds_CDK_Desc_file_path,
            f"{fluorinated_compounds}_Descriptors.csv",
        )
        fluorinated_compounds_CDK_Desc = pd.read_csv(
            fluorinated_compounds_CDK_Desc_path
        )

        # Delete all columns containg 'proton'
        fluorinated_compounds_CDK_Desc = fluorinated_compounds_CDK_Desc.loc[
            :,
            ~fluorinated_compounds_CDK_Desc.columns.str.contains("proton", case=False),
        ]

        # Get index of all F atoms in the fluorinated_compounds molecular
        F_index = fluorinated_compounds_CDK_Desc[
            fluorinated_compounds_CDK_Desc.SMILES == "F"
        ].index

        # Generate a df about NMR peak data, which with F index as index and "NMR_Peaks" as col.
        NMR_Peaks = fluorinated_compounds_NMR_Peaks[
            fluorinated_compounds_NMR_Peaks.Code == fluorinated_compounds
        ]
        NMR_Peaks = NMR_Peaks.reset_index(drop=True)
        NMR_Peaks.index = ["NMR_Peaks"]
        NMR_Peaks = NMR_Peaks.loc[:, F_index]
        NMR_Peaks = NMR_Peaks.T

        NMR_Peaks = NMR_Peaks.rename(lambda x: f"{x}_{fluorinated_compounds}")

        # Extract single values for the 'SMILES', 'Compound name', and 'Solvent_used_for_NMR' columns
        NMR_Peaks["SMILES"] = fluorinated_compounds_NMR_Peaks[
            fluorinated_compounds_NMR_Peaks.Code == fluorinated_compounds
        ]["SMILES"].values[0]
        NMR_Peaks["Compound name"] = fluorinated_compounds_NMR_Peaks[
            fluorinated_compounds_NMR_Peaks.Code == fluorinated_compounds
        ]["Compound name"].values[0]
        NMR_Peaks["Solvent_used_for_NMR"] = fluorinated_compounds_NMR_Peaks[
            fluorinated_compounds_NMR_Peaks.Code == fluorinated_compounds
        ]["Solvent_used_for_NMR"].values[0]

        all_NMR_Peaks = pd.concat([all_NMR_Peaks, NMR_Peaks], axis=0)

    return all_NMR_Peaks


def testModelPerformance_XGBoost_3DNeighborAtoms(
    best_model_file_path, columns_file_path, neighbor_num, test_dataset
):
    """
    Function
    ----------
    Applies the optimized model to the validation or test dataset, outputs the prediction results in a DataFrame,
    and visualizes the prediction performance using both a line plot and a scatter plot.

    Parameters
    ----------
    best_model_file_path: Path to the .json file containing the optimized model.

    columns_file_path: Path to the .pkl file containing the column names of the dataset used for modeling.

    test_dataset: A DataFrame containing information about fluorinated compounds and their corresponding 19F NMR shift values.

    Output
    ----------
    results_table: A DataFrame containing the prediction results for the test_dataset.
    """
    best_model = XGBRegressor()
    best_model.load_model(best_model_file_path)

    #     with open(encoder_path, 'rb') as file:
    #         encoder = pickle.load(file)

    with open(columns_file_path, "rb") as f:
        train_cols = pickle.load(f)

    #     neighbor_atoms_list = all_neighbor_atoms_list[10-neighbor_num:]

    # Step 1. Combine NMR shift values of F atoms with its features
    fluorinated_compounds_w_Desc = Combine_descriptors(
        test_dataset, neighbor_num=neighbor_num
    )

    #     # Step 2. Encode categorical columns in the validation dataset using the same one-hot-encoder used for the training dataset
    #     encoded = encoder.transform(fluorinated_compounds_w_Desc[neighbor_atoms_list])
    #     df_encoded = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(neighbor_atoms_list))
    #     df_encoded.index = fluorinated_compounds_w_Desc.index
    #     fluorinated_compounds_w_Desc = pd.concat([fluorinated_compounds_w_Desc, df_encoded], axis=1)

    # Step 3. Only keep columns that were used in the dataset for modeling while delete other columns
    fluorinated_compounds_w_Desc = fluorinated_compounds_w_Desc[train_cols]

    # For validation purpose, drop rows with NaN values in the 'NMR_Peaks' column
    fluorinated_compounds_w_Desc = fluorinated_compounds_w_Desc.dropna(
        subset=["NMR_Peaks"]
    )

    # Get y values
    y = fluorinated_compounds_w_Desc["NMR_Peaks"]

    # Drop non-numeric columns from X
    #     orig_features = ['SMILES', 'Compound name', 'Solvent_used_for_NMR', 'NMR_Peaks']
    orig_features = ["NMR_Peaks"]
    X = fluorinated_compounds_w_Desc.drop(orig_features, axis=1)

    # Ensure all values in the X are numerical values
    X = X.apply(pd.to_numeric)

    results_table = common.get_results_table(best_model=best_model, X=X, y=y)

    common.plot_prediction_performance(results_table, figure_title=None)

    common.show_results_scatter(results_table, figure_title=None)

    return results_table
