from collections import defaultdict
import re



def clean_director_name(name: str):
    """
    A director's name is standardized by converting it to lowercase, removing spaces, hyphens, and periods
    """
    name = name.lower().replace(" ", "").replace("-", "")
    return re.sub(r"\.", "", name)

def create_name_map(df):
    """
    Creates a mapping dictionary where multiple variations of a name will be mapped to a standardized version.
    Keys: Cleaned director names
    Values: Original names

    """
    name_map = defaultdict(set)
    for i, row in df.iterrows():
        director_list = [name.strip() for name in row["director"].split(",")]
        for director in director_list:
            cleaned_name = clean_director_name(director)
            name_map[cleaned_name].add(director)
    return name_map

def map_director_names(df, name_map):
    """
    Maps and replaces director names in the DataFrame with standardized versions using the name map
    """

    # Reverse the name_map for a faster lookup of the standardized names
    reverse_name_map = {}
    for standard_name, original_names in name_map.items():
        for name in original_names:
            reverse_name_map[name] = standard_name

    def apply_mapping(director_names):
        mapped_directors = [
            reverse_name_map.get(name.strip(), name.strip()) for name in director_names.split(",")
        ]
        return ",".join(mapped_directors)

    df['director'] = df['director'].apply(apply_mapping)