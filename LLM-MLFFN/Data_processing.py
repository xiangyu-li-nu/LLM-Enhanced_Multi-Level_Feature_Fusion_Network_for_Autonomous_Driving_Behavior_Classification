# import pandas as pd
# import re
#
# # 1. Read the data
# df = pd.read_csv('extracted_features_with_analysis11.csv', encoding='latin1')
#
# # 2. Extract the necessary columns
# text_col = 'analysis'
# label_col = 'label'
#
# # 3. Define the list of target words to be removed (including case variations)
# target_words = ['Aggressive', 'Assertive', 'Conservative', 'Moderate']
#
# 4. Create a regular expression pattern, ignoring case
#    \b ensures that only complete words are matched
# pattern = re.compile(r'\b(' + '|'.join(target_words) + r')\b', re.IGNORECASE)
#
# def replace_with_label(text, label):
#     """
#     Remove the target words from the text.
#
#     Parameters:
#     text (str): The original text.
#
#     Returns:
#     str: The text with target words removed.
#     """
#     return pattern.sub(label, text)
#
# # 5. Apply the removal function to each row in the 'analysis' column
# df['new_analysis'] = df.apply(lambda row: replace_with_label(row[text_col], row[label_col]), axis=1)
#
# # 6. Save the results to a new CSV file
# df.to_csv('extracted_features_with_analysis.csv', index=False, encoding='latin1')
#
# print("Removal completed, new analysis has been saved to 'extracted_features_with_analysis.csv'。")
import pandas as pd
import re
import os

# 1. Read the data
# Use absolute path to ensure the file path is correct. Modify the path according to actual situation.
file_path = r'extracted_features_with_analysis11.csv'

# Check if the file exists
if not os.path.exists(file_path):
    print(f"File does not exist: {file_path}")
    exit(1)

try:
    df = pd.read_csv(file_path, encoding='latin1')
    print("File read successfully, data preview:")
    print(df.head())
except pd.errors.EmptyDataError:
    print(f"The file is empty or has no parseable columns: {file_path}")
    exit(1)
except Exception as e:
    print(f"Error occurred while reading the file: {e}")
    exit(1)

# 2. Extract the necessary columns
text_col = 'analysis'
label_col = 'label'

# Check if necessary columns exist
if text_col not in df.columns or label_col not in df.columns:
    print(f"Missing necessary columns: must contain '{text_col}' and '{label_col}' columns.")
    exit(1)

# 3. Define the list of target words to be removed (including case variations)
target_words = ['Aggressive', 'Assertive', 'Conservative', 'Moderate']

# 4. Create a regular expression pattern, ignoring case
#    \b ensures that only complete words are matched
pattern = re.compile(r'\b(' + '|'.join(target_words) + r')\b', re.IGNORECASE)

def remove_target_words(text):
    """
    Remove the target words from the text.

    Parameters:
    text (str): The original text.

    Returns:
    str: The text with target words removed.
    """
    # Use regular expression to replace target words with an empty string
    cleaned_text = pattern.sub('', text)
    # Remove extra spaces
    cleaned_text = re.sub(' +', ' ', cleaned_text).strip()
    return cleaned_text

# 5. Apply the removal function to each row in the 'analysis' column
df['new_analysis'] = df[text_col].apply(remove_target_words)

# 6. Save the results to a new CSV file
output_path = r'shanchu.csv'

try:
    df.to_csv(output_path, index=False, encoding='latin1')
    print(f"Removal completed, new analysis has been saved to '{output_path}'.")
except Exception as e:
    print(f"Error occurred while saving the file: {e}")
