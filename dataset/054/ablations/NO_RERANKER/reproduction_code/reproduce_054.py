import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import tensorflow_decision_forests as tfdf

BASE_PATH = 'https://kdd.ics.uci.edu/databases/census-income/census-income'
train_data = pd.read_csv(f'{BASE_PATH}.data.gz', header=None)
test_data = pd.read_csv(f'{BASE_PATH}.test.gz', header=None)

TARGET_COLUMN_NAME = 'income_level'
train_data[TARGET_COLUMN_NAME] = train_data[TARGET_COLUMN_NAME].map({' - 50000.': 0, ' 50000+.': 1})
test_data[TARGET_COLUMN_NAME] = test_data[TARGET_COLUMN_NAME].map({' - 50000.': 0, ' 50000+.': 1})

NUMERIC_FEATURE_NAMES = ['age', 'wage_per_hour', 'capital_gains', 'capital_losses', 'dividends_from_stocks', 'num_persons_worked_for_employer', 'weeks_worked_in_year']
CATEGORICAL_FEATURE_NAMES = ['class_of_worker', 'detailed_industry_recode', 'detailed_occupation_recode', 'education', 'enroll_in_edu_inst_last_wk', 'marital_stat', 'major_industry_code', 'major_occupation_code', 'race', 'hispanic_origin', 'sex', 'member_of_a_labor_union', 'reason_for_unemployment', 'full_or_part_time_employment_stat', 'tax_filer_stat', 'region_of_previous_residence', 'state_of_previous_residence', 'detailed_household_and_family_stat', 'detailed_household_summary_in_household', 'migration_code-change_in_msa', 'migration_code-change_in_reg', 'migration_code-move_within_reg', 'live_in_this_house_1_year_ago', 'migration_prev_res_in_sunbelt', 'family_members_under_18', 'country_of_birth_father', 'country_of_birth_mother', 'country_of_birth_self', 'citizenship', 'own_business_or_self_employed', 'fill_inc_questionnaire_for_veteran\'s_admin', 'veterans_benefits', 'year']

train_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(train_data, label=TARGET_COLUMN_NAME)
test_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(test_data, label=TARGET_COLUMN_NAME)

gbt_model = tfdf.keras.GradientBoostedTreesModel(task=tfdf.keras.Task.CLASSIFICATION)
gbt_model.compile(metrics=[keras.metrics.BinaryAccuracy(name='accuracy')])
gbt_model.fit(train_dataset, epochs=20, validation_data=test_dataset)
_, accuracy = gbt_model.evaluate(test_dataset, verbose=0)
print(f'Test accuracy: {round(accuracy * 100, 2)}%')