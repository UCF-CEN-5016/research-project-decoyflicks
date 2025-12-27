import math
import urllib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_decision_forests as tfdf

BASE_PATH = "https://kdd.ics.uci.edu/databases/census-income/census-income"
CSV_HEADER = [
    l.decode("utf-8").split(":")[0].replace(" ", "_")
    for l in urllib.request.urlopen(f"{BASE_PATH}.names")
    if not l.startswith(b"|")
][2:]
CSV_HEADER.append("income_level")

train_data = pd.read_csv(
    f"{BASE_PATH}.data.gz",
    header=None,
    names=CSV_HEADER,
)
test_data = pd.read_csv(
    f"{BASE_PATH}.test.gz",
    header=None,
    names=CSV_HEADER,
)

TARGET_COLUMN_NAME = "income_level"
TARGET_LABELS = [" - 50000.", " 50000+."]
NUMERIC_FEATURE_NAMES = [
    "age",
    "wage_per_hour",
    "capital_gains",
    "capital_losses",
    "dividends_from_stocks",
    "num_persons_worked_for_employer",
    "weeks_worked_in_year",
]
CATEGORICAL_FEATURE_NAMES = [
    "class_of_worker",
    "detailed_industry_recode",
    "detailed_occupation_recode",
    "education",
    "enroll_in_edu_inst_last_wk",
    "marital_stat",
    "major_industry_code",
    "major_occupation_code",
    "race",
    "hispanic_origin",
    "sex",
    "member_of_a_labor_union",
    "reason_for_unemployment",
    "full_or_part_time_employment_stat",
    "tax_filer_stat",
    "region_of_previous_residence",
    "state_of_previous_residence",
    "detailed_household_and_family_stat",
    "detailed_household_summary_in_household",
    "migration_code-change_in_msa",
    "migration_code-change_in_reg",
    "migration_code-move_within_reg",
    "live_in_this_house_1_year_ago",
    "migration_prev_res_in_sunbelt",
    "family_members_under_18",
    "country_of_birth_father",
    "country_of_birth_mother",
    "country_of_birth_self",
    "citizenship",
    "own_business_or_self_employed",
    "fill_inc_questionnaire_for_veteran's_admin",
    "veterans_benefits",
    "year",
]

def prepare_dataframe(dataframe):
    dataframe[TARGET_COLUMN_NAME] = dataframe[TARGET_COLUMN_NAME].map(
        TARGET_LABELS.index
    )
    for feature_name in CATEGORICAL_FEATURE_NAMES:
        dataframe[feature_name] = dataframe[feature_name].astype(str)

prepare_dataframe(train_data)
prepare_dataframe(test_data)

print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

NUM_TREES = 250
MIN_EXAMPLES = 6
MAX_DEPTH = 5
SUBSAMPLE = 0.65
VALIDATION_RATIO = 0.1

def specify_feature_usages():
    feature_usages = []
    for feature_name in NUMERIC_FEATURE_NAMES:
        feature_usage = tfdf.keras.FeatureUsage(
            name=feature_name, semantic=tfdf.keras.FeatureSemantic.NUMERICAL
        )
        feature_usages.append(feature_usage)
    for feature_name in CATEGORICAL_FEATURE_NAMES:
        feature_usage = tfdf.keras.FeatureUsage(
            name=feature_name, semantic=tfdf.keras.FeatureSemantic.CATEGORICAL
        )
        feature_usages.append(feature_usage)
    return feature_usages

def create_gbt_model():
    gbt_model = tfdf.keras.GradientBoostedTreesModel(
        features=specify_feature_usages(),
        exclude_non_specified_features=True,
        num_trees=NUM_TREES,
        max_depth=MAX_DEPTH,
        min_examples=MIN_EXAMPLES,
        subsample=SUBSAMPLE,
        validation_ratio=VALIDATION_RATIO,
        task=tfdf.keras.Task.CLASSIFICATION,
    )
    gbt_model.compile(metrics=[keras.metrics.BinaryAccuracy(name="accuracy")])
    return gbt_model

gbt_model = create_gbt_model()

def run_experiment(model, train_data, test_data, num_epochs=1, batch_size=None):
    train_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(
        train_data, label=TARGET_COLUMN_NAME
    )
    test_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(
        test_data, label=TARGET_COLUMN_NAME
    )
    model.fit(train_dataset, epochs=num_epochs, batch_size=batch_size)
    _, accuracy = model.evaluate(test_dataset, verbose=0)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

run_experiment(gbt_model, train_data, test_data)

# Potential typo reproduction
level2_dau_2 = None  # This variable is declared but never used
# Correcting the function call to use level2_dau_2 instead of level3_dau_2
skff_ = selective_kernel_feature_fusion(level1_dau_2, level2_dau_2, level3_dau_2)  # Bug reproduction logic preserved