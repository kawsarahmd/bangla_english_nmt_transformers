import argparse
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from datasets import DatasetDict, Dataset
from huggingface_hub import login

def process_and_push_dataset(input_file, output_file, dataset_name, train_size, val_size, test_size, hf_token):
    try:
        # Read the input CSV file
        data = pd.read_csv(input_file)

        # Filter only the desired columns
        filtered_data = data[["id", "news_id", "bn_text", "en_text"]]

        # Remove rows with null values
        initial_row_count = len(filtered_data)
        filtered_data = filtered_data.dropna()
        rows_removed = initial_row_count - len(filtered_data)
        print(f"Removed {rows_removed} rows with null values. Remaining rows: {len(filtered_data)}")

        # Shuffle the data
        shuffled_data = shuffle(filtered_data, random_state=42)
        print(f"Shuffled dataset rows: {len(shuffled_data)}")

        # Create new columns for input_text and output_text
        en_to_bn = shuffled_data.assign(
            input_text=shuffled_data["en_text"],
            output_text=shuffled_data["bn_text"]
        )

        bn_to_en = shuffled_data.assign(
            input_text=shuffled_data["bn_text"],
            output_text=shuffled_data["en_text"]
        )

        # Combine both directions into one DataFrame
        final_data = pd.concat([en_to_bn, bn_to_en], ignore_index=True)

        # Drop the original `bn_text` and `en_text` columns
        final_data = final_data.drop(columns=["bn_text", "en_text"])

        # Shuffle the combined data to ensure a mix of pairs
        final_data = shuffle(final_data, random_state=42).reset_index(drop=True)

        # Save the final dataset locally
        final_data.to_csv(output_file, index=False)
        print(f"Processed dataset saved at: {output_file}")
        print(f"Final dataset rows: {len(final_data)}")

        # Split the dataset into train, validation, and test
        train_data, temp_data = train_test_split(
            final_data, test_size=(1 - train_size), random_state=42
        )
        val_size_adjusted = val_size / (val_size + test_size)
        val_data, test_data = train_test_split(
            temp_data, test_size=(1 - val_size_adjusted), random_state=42
        )

        # Report split sizes
        print(f"Train split rows: {len(train_data)}")
        print(f"Validation split rows: {len(val_data)}")
        print(f"Test split rows: {len(test_data)}")

        # Convert splits to Hugging Face dataset format
        dataset_dict = DatasetDict({
            "train": Dataset.from_pandas(train_data.reset_index(drop=True)),
            "validation": Dataset.from_pandas(val_data.reset_index(drop=True)),
            "test": Dataset.from_pandas(test_data.reset_index(drop=True))
        })

        # Login to Hugging Face Hub
        login(hf_token)

        # Push the dataset to Hugging Face
        dataset_dict.push_to_hub(dataset_name)
        print(f"Dataset {dataset_name} successfully uploaded to Hugging Face!")
    except FileNotFoundError:
        print(f"File not found: {input_file}")
    except KeyError as e:
        print(f"Column missing: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Process, save, split, and push dataset to Hugging Face Hub.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the processed CSV file")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the Hugging Face dataset")
    parser.add_argument("--train_size", type=float, default=0.9, help="Proportion of the training set")
    parser.add_argument("--val_size", type=float, default=0.08, help="Proportion of the validation set")
    parser.add_argument("--test_size", type=float, default=0.02, help="Proportion of the test set")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face authentication token")
    args = parser.parse_args()

    # Validate splits
    if not abs(args.train_size + args.val_size + args.test_size - 1.0) < 1e-6:
        raise ValueError("Train, validation, and test sizes must sum to 1.")

    # Run the processing function
    process_and_push_dataset(
        args.input_file,
        args.output_file,
        args.dataset_name,
        args.train_size,
        args.val_size,
        args.test_size,
        args.hf_token
    )
