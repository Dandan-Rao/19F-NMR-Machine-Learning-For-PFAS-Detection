import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import math


from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import KNNImputer
from xgboost import XGBRegressor

random_state = 1

from rdkit import Chem
import pickle

import atomic_features_3D
import hose_code

random_state = 1
pd.set_option("display.max_rows", 10)
pd.set_option("display.max_columns", 100)


def convert_column_name(name):
    """
    Convert column name to integer if possible, otherwise keep it as is.
    """
    try:
        return int(name)
    except ValueError:
        return name


def canonical_smiles(smiles):
    """
    Convert SMILES to canonical SMILES
    """
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    smiles = [Chem.MolToSmiles(mol) for mol in mols]
    return smiles


def plot_prediction_performance(results_table, figure_title=None):
    """
    Plot the distribution of the prediction error of a model.
    """
    num_below_1, num_below_2, num_below_3, num_below_5, num_below_10, num_below_20 = (
        0,
        0,
        0,
        0,
        0,
        0,
    )
    total_count = len(results_table["diff"])

    for i in results_table["diff"]:
        if i < 20:
            num_below_20 += 1
        if i < 10:
            num_below_10 += 1
        if i < 5:
            num_below_5 += 1
        if i < 3:
            num_below_3 += 1
        if i < 2:
            num_below_2 += 1
        if i < 1:
            num_below_1 += 1

    x = [0, 1, 2, 3, 5, 10, 20]
    y = [
        0,
        num_below_1,
        num_below_2,
        num_below_3,
        num_below_5,
        num_below_10,
        num_below_20,
    ]
    y = [val / total_count for val in y]

    # Create the figure and axes with specified size
    cm = 1 / 2.54  # centimeters in inches
    fig, ax = plt.subplots(figsize=(10 * cm, 9 * cm))  # 8 cm x 8 cm

    ax.plot(x, y, marker="o", color="#069AF3")
    ax.set_title(figure_title)
    #     plt.grid(True)
    ax.set_xlim([-1, 22])

    # Set border (edge) line width
    ax.spines["top"].set_linewidth(1)  # Top border
    ax.spines["right"].set_linewidth(1)  # Right border
    ax.spines["bottom"].set_linewidth(1)  # Bottom border
    ax.spines["left"].set_linewidth(1)  # Left border

    # Set axis titles and tick label font sizes
    ax.set_xlabel("Threshold (ppm)", fontsize=14)  # Replace with your label
    ax.set_ylabel("Cumulative Proportion", fontsize=14)  # Replace with your label
    ax.tick_params(axis="x", labelsize=12)  # X-axis numbers font size
    ax.tick_params(axis="y", labelsize=12)  # Y-axis numbers font size

    # Add annotations to each data point
    for i in range(len(x)):
        ax.annotate(
            f"({x[i]}, {y[i]:.2f})",
            (x[i], y[i]),
            textcoords="offset points",
            xytext=(0, 5),
            ha="left",
        )

    plt.show()


def show_results_scatter(results_table, figure_title=None):
    """
    Plot the scatter plot of actual vs predicted values.
    """
    results_table = results_table.dropna()
    r2 = r2_score(results_table["actual"], results_table["prediction"])

    # Create the figure and axes with specified size
    cm = 1 / 2.54  # centimeters in inches
    fig, ax = plt.subplots(figsize=(10 * cm, 9 * cm))

    ax.scatter(
        x=results_table["actual"],
        y=results_table["prediction"],
        alpha=0.6,
        color="#069AF3",
    )

    ax.plot([30, -300], [30, -300], c="red")
    ax.plot([30, -300], [40, -290], c="green", linestyle="dashed")
    ax.plot([30, -300], [20, -310], c="green", linestyle="dashed")

    ax.set_ylim([-280, 30])
    ax.set_xlim([-280, 30])
    ax.set_title(figure_title)
    print(f"R2 = {r2:.2f}")

    # Set border (edge) line width
    ax.spines["top"].set_linewidth(1)  # Top border
    ax.spines["right"].set_linewidth(1)  # Right border
    ax.spines["bottom"].set_linewidth(1)  # Bottom border
    ax.spines["left"].set_linewidth(1)  # Left border

    # Set axis titles and tick label font sizes
    ax.set_xlabel("Actual (ppm)", fontsize=14)  # Replace with your label
    ax.set_ylabel("Prediction (ppm)", fontsize=14)  # Replace with your label
    ax.tick_params(axis="x", labelsize=12)  # X-axis numbers font size
    ax.tick_params(axis="y", labelsize=12)  # Y-axis numbers font size

    mse = mean_squared_error(results_table["actual"], results_table["prediction"])
    rmse = math.sqrt(mse)
    print(f"RMSE = {rmse:.2f}")
    mae = mean_absolute_error(results_table["actual"], results_table["prediction"])
    print(f"MAE = {mae}")
    plt.show()


def one_hot_encoding_of_smiles_neighborN(
    dataset,
    smiles_neighbors=[
        "SMILES_neighbor5",
        "SMILES_neighbor4",
        "SMILES_neighbor3",
        "SMILES_neighbor2",
        "SMILES_neighbor1",
    ],
):
    """
    One hot encoding of the categorical values in columns smiles_neighbors
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    cols = smiles_neighbors
    # Fit the encoder to the categorical column(s)
    encoder.fit(dataset[cols])
    encoded = encoder.transform(dataset[cols])

    # Create a DataFrame with the encoded data
    df_encoded = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cols))
    df_encoded.index = dataset.index

    temp = dataset.drop(columns=cols)

    # Combine the encoded data with the rest of the DataFrame
    dataset_encoded = pd.concat([temp, df_encoded], axis=1)

    print(f"-----Shape of the dataset after encoding the categorical values------")
    print(dataset_encoded.shape)
    return dataset_encoded, encoder


def drop_smiles_neighborN(
    dataset,
    smiles_neighbors=[
        "SMILES_neighbor5",
        "SMILES_neighbor4",
        "SMILES_neighbor3",
        "SMILES_neighbor2",
        "SMILES_neighbor1",
    ],
):
    """
    Drop the categorical values in columns smiles_neighbors
    """
    #     dataset_dropCat = dataset.drop(smiles_neighbors, axis = 1)
    #     print(dataset_dropCat.shape)
    encoded_columns = [
        col for col in dataset.columns if any(orig in col for orig in smiles_neighbors)
    ]

    return dataset.drop(columns=encoded_columns)


# Data processing
def drop_constant_col(df):
    """
    Drop constant columns
    """
    df = df.loc[:, df.nunique() > 1]
    return df


def find_bool_columns(df):
    """
    Identify bool columns
    """
    bool_columns = df.select_dtypes(exclude=["number"]).sum()
    print("Non-Numeric Columns:")
    print(bool_columns)


def count_non_numeric_values(df):
    """
    Count non-numeric values in each columns
    """
    non_numeric_counts = df.map(lambda x: not isinstance(x, (float, int))).sum()

    # Filter to show only columns with non-numeric values more than 0
    filtered_non_numeric_counts = non_numeric_counts[non_numeric_counts > 0]

    print("Columns with Non-Numeric Values:")
    print(filtered_non_numeric_counts)


def convert_to_numeric(value):
    """
    Convert value to numeric if possible
    """
    try:
        return pd.to_numeric(value)
    except:
        return np.nan


def drop_low_cv_cols(df):
    """
    Drop columns with <2% cv
    """
    cv = {}
    for col in df.columns:
        series = df[col]
        std = np.std(series, ddof=1)
        mean = np.mean(series)
        cv[col] = std / mean * 100

    cv_df = pd.DataFrame.from_dict(cv, orient="index")

    print(f"Number of columns with <2% cv: {cv_df[cv_df[0] < 2].shape[0]}")

    cv_df[0].hist(bins=20, range=(0, 100))
    plt.xlabel("CV (%)")
    plt.ylabel("Number of Columns")

    ## Let's drop columns with cv < 2%
    low_cv_cols = cv_df[cv_df[0] < 2].index
    df_new = df.drop(low_cv_cols, axis=1)
    return df_new


def drop_high_ratio_NaN_cols(df):
    """
    Drop columns with >80% NaN values
    """
    bar = 0.8 * df.shape[0]
    df_cleaned = df.loc[:, df.isnull().sum() < bar]
    return df_cleaned


def fill_NaN(X):
    """
    Fill NaN values using KNN imputer
    """
    imputer = KNNImputer(n_neighbors=2)
    X_filled = imputer.fit_transform(X)
    X_filled = pd.DataFrame(X_filled, columns=X.columns, index=X.index)
    return X_filled, imputer


def find_highly_correlated_features(df, threshold=0.99):
    """
    Find highly correlated features in a DataFrame.
    """
    correlated_cols = {}
    corr = df.corr()

    for col in df.columns:
        correlated_features = corr.index[corr[col] > threshold].tolist()
        correlated_features.remove(col)  # Remove the feature itself from the list
        if correlated_features:
            correlated_cols[col] = correlated_features
    return correlated_cols


def delete_highly_correlated_features(df, dictionary):
    """
    Delete highly correlated features in a DataFrame.
    """
    all_col = df.columns

    for col in all_col:
        if col in df.columns and col in dictionary:
            cols_to_delete = dictionary[col]
            df = df.drop(cols_to_delete, axis=1)

    return df


def drop_categorical_columns(df):
    """
    Drop categorical columns
    """
    # Identify categorical columns (either of type 'object' or 'category')
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Drop the categorical columns from the DataFrame
    df_no_categorical = df.drop(columns=categorical_cols)

    return df_no_categorical


def get_results_table(best_model, X, y):
    """
    Get the results table for a model. It contains the actual values, predicted values, and the absolute difference between them.
    """
    y_predict = best_model.predict(X)
    compare = pd.DataFrame(y)
    compare.columns = ["actual"]
    compare["prediction"] = y_predict

    compare["actual"] = pd.to_numeric(compare["actual"])
    compare["prediction"] = pd.to_numeric(compare["prediction"])

    compare["diff"] = compare["actual"] - compare["prediction"]
    compare["diff"] = compare["diff"].abs()
    return compare


def testRidgeCVPerformance(
    dataset, neighbor_num, RidgeCVmodel_path, scaler_path, imputer_path, columns_path
):
    """
    Test the performance of Ridge model. Ouput the results table, plot the error distribution of the model, and plot the scatter plot of actual vs predicted values.
    """
    with open(RidgeCVmodel_path, "rb") as file:
        best_model = pickle.load(file)

    with open(scaler_path, "rb") as file:
        scaler = pickle.load(file)

    with open(imputer_path, "rb") as file:
        imputer = pickle.load(file)

    with open(columns_path, "rb") as file:
        train_columns = pickle.load(file)

    # convert values to numeric values when possible
    dataset = atomic_features_3D.Combine_descriptors(
        dataset, neighbor_num=neighbor_num, with_additional_info=True
    )
    #     dataset = dataset.rename_axis('atomCode_fluorinated_compoundsCode', inplace = True)
    dataset.apply(convert_to_numeric)

    # drop rows with NaN values in the 'NMR_Peaks' column
    dataset_dropNaN = dataset.dropna(subset=["NMR_Peaks"])

    dataset_dropNaN = dataset_dropNaN[train_columns]
    dataset_dropNaN_imputed = imputer.transform(dataset_dropNaN)

    dataset_dropNaN_imputed = pd.DataFrame(
        dataset_dropNaN_imputed,
        columns=dataset_dropNaN.columns,
        index=dataset_dropNaN.index,
    )

    y = dataset_dropNaN_imputed["NMR_Peaks"]
    X = dataset_dropNaN_imputed.drop(["NMR_Peaks"], axis=1)

    X_scaled = scaler.transform(X)

    X_scaled = pd.DataFrame(X_scaled)
    X_scaled.columns = X.columns
    X_scaled.index = X.index

    results_table = get_results_table(best_model=best_model, X=X_scaled, y=y)
    plot_prediction_performance(results_table, figure_title=None)
    show_results_scatter(results_table, figure_title=None)
    return results_table


def get_XGBoost_model_results(
    best_model_file_path, columns_file_path, fluorinated_compounds_w_Desc
):
    """
    Get the results table for a XGBoost model. It contains the actual values, predicted values, and the absolute difference between them.
    Plot the error distribution of the model, and plot the scatter plot of actual vs predicted values.
    """

    best_model = XGBRegressor()
    best_model.load_model(best_model_file_path)

    with open(columns_file_path, "rb") as f:
        train_cols = pickle.load(f)

    # Step 1. Only keep columns that were used in the dataset for modeling while delete other columns
    fluorinated_compounds_w_Desc = fluorinated_compounds_w_Desc[train_cols]

    # Get y values
    y = fluorinated_compounds_w_Desc["NMR_Peaks"]

    orig_features = ["NMR_Peaks"]
    X = fluorinated_compounds_w_Desc.drop(orig_features, axis=1)

    # Ensure all values in the X are numerical values
    X = X.apply(pd.to_numeric)

    results_table = get_results_table(best_model=best_model, X=X, y=y)

    return results_table


def safe_split(index_value):
    """
    Split the index value into two parts based on the first underscore.
    """
    parts = index_value.split("_", 1)
    if len(parts) == 2:
        return parts
    elif len(parts) == 1:
        return [parts[0], None]  # Only one part, return None for the second
    else:
        return [None, None]  # No parts, return None for both


from IPython.display import display, HTML, Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def display_results(results):
    """
    Display the results of the prediction.
    It contains the error distribution and the scatter plot of actual vs predicted values.
    """
    # Set style for better visualization
    plt.style.use("seaborn-v0_8-darkgrid")

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4), width_ratios=[3, 1])
    fig.subplots_adjust(hspace=0.1)

    # Prepare data
    real_PFAS_spectra_df = results[["fluorinated_compounds", "atom_index", "actual"]]
    code = real_PFAS_spectra_df["fluorinated_compounds"][0]
    real_PFAS_spectra_df = real_PFAS_spectra_df.pivot(
        index="fluorinated_compounds", columns="atom_index", values="actual"
    )

    predicted_PFAS_spectra_df = results[
        ["fluorinated_compounds", "atom_index", "ensembeled_model"]
    ]
    predicted_PFAS_spectra_df = predicted_PFAS_spectra_df.pivot(
        index="fluorinated_compounds", columns="atom_index", values="ensembeled_model"
    )

    # Colors for better contrast
    actual_color = "g"  # Green
    predict_color = "C9"  # Blue

    # Get actual and predicted value counts
    actual = real_PFAS_spectra_df.loc[code, :].value_counts()
    predict = predicted_PFAS_spectra_df.loc[code, :].value_counts()

    # Plot in the first subplot (ax1)
    if actual.empty:
        plt.vlines(
            predict.index, ymin=0, ymax=-predict.values, color="w", label="Actual"
        )

    else:
        ax1.vlines(
            actual.index,
            ymin=0,
            ymax=-actual.values,
            color=actual_color,
            label="Actual",
            linewidth=2,
        )

    # Plot predicted values
    ax1.vlines(
        predict.index,
        ymin=0,
        ymax=predict.values,
        color=predict_color,
        label="Prediction",
        linewidth=2,
    )

    # Add similarity levels
    #     temp = results[results['fluorinated_compounds'] == 'temp']
    for i, j in zip(predict.index, predict.values):
        similarity_level = results[results["ensembeled_model"] == i][
            "similarity_levels"
        ]
        if not similarity_level.empty:
            ax1.text(
                i,
                j + 0.05,
                similarity_level.iloc[0],
                ha="center",
                va="bottom",
                fontsize=10,
                color="black",
                bbox=dict(facecolor="white", alpha=0.3),
            )

    # Set plot limits and labels
    if not actual.empty:
        x_min = min(actual.index.min(), predict.index.min()) - 10
        x_max = max(actual.index.max(), predict.index.max()) + 10
        y_max = max(actual.values.max(), predict.values.max()) + 0.2
    else:
        x_min = predict.index.min() - 10
        x_max = predict.index.max() + 10
        y_max = predict.values.max() + 0.2
        ax1.text(
            (x_min + x_max) / 2,
            -y_max / 2,
            "No report value",
            color=actual_color,
            ha="center",
            va="center",
            fontsize=17,
        )

    ax1.set_xlim([x_min, x_max])
    ax1.set_ylim([-y_max, y_max])

    # Add labels
    ax1.text(
        x_min - 2,
        y_max / 2,
        "Prediction",
        color=predict_color,
        rotation=90,
        ha="center",
        va="center",
        fontsize=14,
    )
    ax1.text(
        x_min - 2,
        -y_max / 2,
        "Report",
        color=actual_color,
        rotation=90,
        ha="center",
        va="center",
        fontsize=14,
    )

    ax1.set_xlabel(r"$^{19}$F NMR shift", fontsize=12)
    ax1.set_yticks([])
    ax1.axhline(0, color="k", linewidth=0.5)
    ax1.set_title("NMR Shift Prediction Results", fontsize=14, pad=20)

    # Add confidence level information in the second subplot (ax2)
    confidence_data = {
        "Level": [6, 5, 4, 3, 2, 1],
        "Error": [0.89, 1.05, 1.53, 5.05, 6.91, 11.88],
    }

    ax2.axis("off")
    ax2.set_xlim(0, 2)  # Adjust limits as needed
    ax2.set_ylim(0, 2)
    table_text = (
        "Note:\n"
        + "The number above prediction results indicates\n"
        + "the confidence level of the prediction.\n\n"
        + "Confidence Levels (75% prediction error):\n\n"
    )
    for level, error in zip(confidence_data["Level"], confidence_data["Error"]):
        table_text += f"Level {level}: ±{error} ppm\n"
    ax2.text(
        0,
        1,
        table_text,
        ha="left",
        va="center",
        fontsize=12,
        bbox=dict(facecolor="white", edgecolor="white", alpha=0.9),
    )

    plt.show()

    details = results[
        [
            "atom_index",
            "similarity_levels",
            "actual",
            "ensembeled_model",
            "ensembeled_model_error",
        ]
    ].rename(
        columns={
            "atom_index": "Atom Index",
            "similarity_levels": "Confidence Level",
            "actual": "Report Values",
            "ensembeled_model": "Prediction Results",
            "ensembeled_model_error": "Prediction Error",
        }
    )
    details.reset_index(drop=True, inplace=True)
    # Display molecular structure if available
    file_path = os.path.join("..", "app", 'temp')
    if os.path.exists(os.path.join(file_path, "temp.png")):
        filepath = os.path.join(file_path, "temp.png")
        display(Image(filepath))

    # Display detailed results in a styled table
    # Style the dataframe
    styled_details = (
        details.style.set_properties(**{"text-align": "center"})
        .format({"Prediction Results": "{:.2f}", "Prediction Error": "{:.2f}"})
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#f8f9fa"),
                        ("color", "black"),
                        ("font-weight", "bold"),
                        ("text-align", "center"),
                        ("padding", "8px"),
                    ],
                },
                {"selector": "td", "props": [("padding", "8px")]},
            ]
        )
    )

    display(styled_details)


# -


def predictor(
    smiles,
    train_fluorinated_compounds_file_path=os.path.join(
        "..", "dataset", "Raw_PFAS 19F NMR spectra data.csv"
    ),
    HOSE_Code_database_file_path=os.path.join(
        "..", "artifacts", "temp", "HOSE_database_all_fluorianted_compounds.csv"
    ),
    best_XGBoost_mode_file_path=os.path.join(
        "..",
        "artifacts",
        "models",
        "Final_xgboost_model_3D_descriptors_n5_full_dataset_Random_Search.json",
    ),
):
    """
    Apply the trained XGBoost model and the HOSE code method to predict the 19F NMR shift a molecule given its SMILES representation.
    """
    # Generate sdf file from SMILES
    atomic_features_3D.get_sdf_file(smiles)
    # Generate CDK descriptors and Neighbors information
    atomic_features_3D.get_descriptors_and_neighbors_info()

    # Use the CDK descriptors and Neighbors information to get the features table
    train_dataset = pd.read_csv(train_fluorinated_compounds_file_path, index_col=0)
    dataset = atomic_features_3D.get_test_fluorianted_compounds_info(
        smiles, train_dataset
    )

    NMR_peaks_with_desc = atomic_features_3D.get_features_table(dataset)

    # Get Prediction results from HOSE model
    HOSE_results = hose_code.get_HOSE_prediction_results_table(
        HOSE_Code_database_file_path, dataset
    )

    # Get Prediction results from XGBoost model
    XGBoost_results = get_XGBoost_model_results(
        best_model_file_path=best_XGBoost_mode_file_path,
        columns_file_path=os.path.join(
            "..", "artifacts", "models", "column_names_neighbor5_xgboost.pkl"
        ),
        fluorinated_compounds_w_Desc=NMR_peaks_with_desc,
    )
    combined_prediction = pd.DataFrame()
    combined_prediction = HOSE_results.copy()
    combined_prediction.rename(
        columns={"prediction": "HOSE_model_prediction"}, inplace=True
    )
    combined_prediction = combined_prediction[
        ["actual", "similarity_levels", "HOSE_model_prediction"]
    ]
    combined_prediction["XGBoost_model_prediction"] = XGBoost_results["prediction"]

    ensembled_XGBoost_and_HOSE = []
    for i, row in combined_prediction.iterrows():
        if row["similarity_levels"] >= 4:
            ensembled_XGBoost_and_HOSE.append(row["HOSE_model_prediction"])
        else:
            ensembled_XGBoost_and_HOSE.append(row["XGBoost_model_prediction"])

    combined_prediction["ensembeled_model"] = ensembled_XGBoost_and_HOSE
    combined_prediction["ensembeled_model_error"] = (
        combined_prediction["ensembeled_model"] - combined_prediction["actual"]
    )
    combined_prediction["ensembeled_model_error"] = combined_prediction[
        "ensembeled_model_error"
    ].abs()

    ensemble = combined_prediction.drop(
        ["HOSE_model_prediction", "XGBoost_model_prediction"], axis=1
    )

    split_values = [safe_split(idx) for idx in ensemble.index]

    # Create new columns from the split values
    ensemble["atom_index"] = [val[0] for val in split_values]
    ensemble["fluorinated_compounds"] = [val[1] for val in split_values]
    # Usage
    display_results(ensemble)
