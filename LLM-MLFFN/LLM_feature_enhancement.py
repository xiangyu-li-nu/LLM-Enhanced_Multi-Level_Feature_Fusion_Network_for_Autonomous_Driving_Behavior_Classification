import pandas as pd
from tqdm import tqdm
import time
import openai
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Get the API key
# api_key = os.getenv('')
# if not api_key:
#     logging.error("OPENAI_API_KEY environment variable not set. Please set it and try again.")
#     raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# Configure OpenAI client
openai.api_key = ''
openai.api_base = 'https://api.openai.com/v1'  # Use the official API endpoint

# Set the basic parameters for API requests
MODEL = "gpt-4"  # Select the model based on needs
MAX_RETRIES = 5  # Maximum retry attempts
SLEEP_TIME = 2  # Retry wait time (seconds)

# Read CSV file
input_csv = 'extracted_featuresyuanshi.csv'
output_csv = 'extracted_features_with_analysis11.csv'

# Check if the output file already exists to decide whether to write the header
if os.path.exists(output_csv):
    df = pd.read_csv(output_csv)
    start_index = len(df)
    logging.info(f"Detected existing output file, starting index: {start_index}")
else:
    df = pd.read_csv(input_csv)
    df['analysis'] = ""  # Initialize 'analysis' column
    start_index = 0
    logging.info("No existing output file found, initializing new DataFrame.")

# Define a function to generate analysis
def generate_analysis(sample_features):
    # Construct the prompt
    prompt = f"""Reference the following sample:

### Example 1: Aggressive
Feature values:
acceleration_autocorrelation: 0.498585529
acceleration_change_rate: -0.540906302
...
[Other example content omitted for brevity]

Please analyze the driving style based on the following feature values and describe it in natural language (within 100 words):

Feature values:
acceleration_autocorrelation: {sample_features['acceleration_autocorrelation']}
acceleration_change_rate: {sample_features['acceleration_change_rate']}
acceleration_jerk_cross_correlation: {sample_features['acceleration_jerk_cross_correlation']}
acceleration_kurtosis: {sample_features['acceleration_kurtosis']}
acceleration_max: {sample_features['acceleration_max']}
acceleration_mean: {sample_features['acceleration_mean']}
acceleration_median: {sample_features['acceleration_median']}
acceleration_min: {sample_features['acceleration_min']}
acceleration_quantile25: {sample_features['acceleration_quantile25']}
acceleration_quantile75: {sample_features['acceleration_quantile75']}
acceleration_skewness: {sample_features['acceleration_skewness']}
acceleration_std: {sample_features['acceleration_std']}
jerk_kurtosis: {sample_features['jerk_kurtosis']}
jerk_max: {sample_features['jerk_max']}
jerk_mean: {sample_features['jerk_mean']}
jerk_median: {sample_features['jerk_median']}
jerk_min: {sample_features['jerk_min']}
jerk_quantile25: {sample_features['jerk_quantile25']}
jerk_quantile75: {sample_features['jerk_quantile75']}
jerk_skewness: {sample_features['jerk_skewness']}
jerk_std: {sample_features['jerk_std']}
num_hard_accelerations: {sample_features['num_hard_accelerations']}
num_hard_brakes: {sample_features['num_hard_brakes']}
num_hard_turns: {sample_features['num_hard_turns']}
speed_acceleration_cross_correlation: {sample_features['speed_acceleration_cross_correlation']}
speed_autocorrelation: {sample_features['speed_autocorrelation']}
speed_change_rate: {sample_features['speed_change_rate']}
speed_kurtosis: {sample_features['speed_kurtosis']}
speed_max: {sample_features['speed_max']}
speed_mean: {sample_features['speed_mean']}
speed_median: {sample_features['speed_median']}
speed_min: {sample_features['speed_min']}
speed_quantile25: {sample_features['speed_quantile25']}
speed_quantile75: {sample_features['speed_quantile75']}
speed_skewness: {sample_features['speed_skewness']}
speed_std: {sample_features['speed_std']}

Analysis in English:
"""

    for attempt in range(MAX_RETRIES):
        try:
            # Call OpenAI's ChatCompletion API
            completion = openai.ChatCompletion.create(
                model=MODEL,
                messages=[
                    {"role": "system",
                     "content": "You are a driving behavior analyst, skilled in analyzing driving styles based on vehicle sensor data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=500,
                n=1,
                stop=None,
            )
            # Ensure the response structure is correct
            if 'choices' in completion and len(completion['choices']) > 0:
                analysis = completion['choices'][0]['message']['content'].strip()
                return analysis
            else:
                logging.warning(f"Unexpected response structure: {completion}")
                return "Analysis failed: Unexpected response structure"
        except openai.error.OpenAIError as e:
            logging.error(f"OpenAI API request failed: {e}")
            if attempt < MAX_RETRIES - 1:
                logging.info(f"Waiting {SLEEP_TIME} seconds before retrying...")
                time.sleep(SLEEP_TIME)
            else:
                logging.error("Maximum retry attempts reached, skipping this sample.")
                return "Analysis failed"
        except Exception as e:
            logging.error(f"Unexpected error occurred: {e}")
            return "Analysis failed"

    return "Analysis failed"

# Add a new column to store analysis results
if 'analysis' not in df.columns:
    df['analysis'] = ""
    logging.info("Added 'analysis' column to DataFrame.")

# Use tqdm to show the progress bar
total_samples = df.shape[0]
logging.info(f"Starting to process {total_samples} samples.")

for index, row in tqdm(df.iterrows(), total=total_samples, desc="Processing samples", initial=start_index):
    if index < start_index:
        continue  # Skip already processed samples

    # Extract features of the current sample
    sample_features = row.to_dict()
    # Remove 'label' and 'analysis' columns (if they exist)
    sample_features.pop('label', None)
    sample_features.pop('analysis', None)

    # Generate analysis
    analysis = generate_analysis(sample_features)

    logging.info(f"Sample {index}: {analysis}")

    # Save the analysis to the new column
    df.at[index, 'analysis'] = analysis

    # Save the current progress to the output CSV file
    df.to_csv(output_csv, index=False)

    # To avoid triggering API rate limits, apply a proper delay
    time.sleep(1)  # Adjust based on actual conditions

logging.info("Analysis completed. Results have been saved to 'extracted_features_with_analysis1.csv'")
