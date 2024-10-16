import pandas as pd

def read_data(path: str, columns: list[str]) -> pd.DataFrame:
    """Read data from a file into a Pandas DataFrame.

    Args:
        path (str): The path to the file to read.
        columns (list[str]): The column names.

    Returns:
        pd.DataFrame: The DataFrame containing the data.
    """
    data = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            fields = line.strip().split("\t")
            data.append(fields)

    return pd.DataFrame(data, columns=columns)
