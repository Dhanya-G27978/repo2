from sklearn.datasets import load_iris
import pandas as pd


def load_iris_data():
    iris = load_iris()

    df = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )

    df["species"] = [
        iris.target_names[i]
        for i in iris.target
    ]

    return df


def dataset_summary(df):
    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nSpecies Count:")
    print(df["species"].value_counts())

def show_average_measurements(df):
    print("\nAverage Measurements:")
    print(df.drop(columns=["species"]).mean())

def show_species_summary(df):
    """Display average measurements for each Iris species."""
    numeric_columns = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]

    species_summary = (
        df.groupby("species")[numeric_columns]
        .mean()
        .round(2)
    )

    print("\nAverage Measurements by Species:")
    print(species_summary)
    

def main():
    df = load_iris_data()
    dataset_summary(df)
    show_average_measurements(df)
    show_species_summary(df)


if __name__ == "__main__":
    main()