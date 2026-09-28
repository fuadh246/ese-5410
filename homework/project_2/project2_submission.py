"""Project 2 Gradescope submission.

Generated from the Project 2 notebook. This module intentionally contains no
notebook magics, plotting cells, local-grading calls, or dataset downloads.
"""

import numpy as np
import pandas as pd

def create_large_university_mask(df):
    return df["Enroll"] > df["Enroll"].mean()


def get_large_university_enroll_75th_percentile(df):
    return df[df["Enroll"] > df["Enroll"].mean()].describe(include="all")["Enroll"]["75%"]


def calculate_acceptance_rate(df):
    return df["Accept"] / df["Apps"]


def get_highest_population_index(df):
    return df["Population"].idxmax()


def get_lowest_median_income_index(df):
    return df["MedInc"].idxmin()


def get_highest_average_occupancy_index(df):
    return df["AveOccup"].idxmax()


def count_bay_area_districts(df):
    bay_area_mask = df["Latitude"].between(37, 38, inclusive="both") & df["Longitude"].between(-122, -121, inclusive="both")
    return bay_area_mask.sum()


def get_median_rooms_for_high_income(df):
    high_income_mask = df["MedInc"] > 5
    return df.loc[high_income_mask, "AveRooms"].median()


def get_min_house_value_index_above_median_income(df):
    median_income = df["MedInc"].median()
    filtered_df = df[df["MedInc"] > median_income]
    return filtered_df["MedHouseVal"].idxmin()


def count_districts_above_median_rooms_and_value(df):
    median_rooms = df["AveRooms"].median()
    median_house_value = df["MedHouseVal"].median()
    mask = (df["AveRooms"] > median_rooms) & (df["MedHouseVal"] > median_house_value)
    return mask.sum()
