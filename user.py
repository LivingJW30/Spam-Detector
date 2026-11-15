"""
This script was written in one shot by GPT-6 Luna to extract the features from a set of emails that were generated using different models on Arena AI.
The features were then fed to the spam detector model to test its accuracy on this brand new set of randomly generated emails.

These emails are stored along with their features and labels in the emails_scored.json file. The "actual_label" key is the human-analyzed determination of the email, whereas the "label" key 
is what the model predicted. As seen in the JSON file, across the 11 emails, only 2 emails were mislabeled. Emails 8 and 11 were both benign emails but were
incorrectly labeled as spam. Looking closer at the contents of those two emails (one being an order confirmation and the other a sales ad), it is understandable why the model struggled.
"""
import json
import re
import joblib
import pandas as pd
from collections import Counter


# --------------------------------------------------
# 1. Define the 48 words used by Spambase
# --------------------------------------------------

SPAM_WORDS = [
    "make", "address", "all", "3d", "our", "over",
    "remove", "internet", "order", "mail", "receive",
    "will", "people", "report", "addresses", "free",
    "business", "email", "you", "credit", "your",
    "font", "000", "money", "hp", "hpl", "george",
    "650", "lab", "labs", "telnet", "857", "data",
    "415", "85", "technology", "1999", "parts", "pm",
    "direct", "cs", "meeting", "original", "project",
    "re", "edu", "table", "conference"
]

SPECIAL_CHARACTERS = [";", "(", "[", "!", "$", "#"]


# --------------------------------------------------
# 2. Calculate all 57 features for one email
# --------------------------------------------------

def parse_features(body):

    features = {}

    # Extract words using alphanumeric sequences.
    # Lowercase a separate copy so capitalization is preserved
    # in the original body for the capital-run calculations.
    words = re.findall(r"[A-Za-z0-9]+", body.lower())

    total_words = len(words)
    word_counts = Counter(words)

    # Calculate the 48 word frequencies.
    for word in SPAM_WORDS:

        if total_words > 0:
            frequency = (word_counts[word] / total_words) * 100
        else:
            frequency = 0.0

        features[f"word_freq_{word}"] = frequency

    # Calculate the 6 character frequencies.
    # len(body) includes spaces, tabs, and newline characters.
    total_characters = len(body)

    for character in SPECIAL_CHARACTERS:

        if total_characters > 0:
            frequency = (body.count(character) / total_characters) * 100
        else:
            frequency = 0.0

        features[f"char_freq_{character}"] = frequency

    # Find consecutive sequences of uppercase letters.
    capital_runs = re.findall(r"[A-Z]+", body)

    run_lengths = [len(run) for run in capital_runs]

    if run_lengths:
        features["capital_run_length_average"] = (
            sum(run_lengths) / len(run_lengths)
        )
        features["capital_run_length_longest"] = max(run_lengths)
        features["capital_run_length_total"] = sum(run_lengths)

    else:
        features["capital_run_length_average"] = 0.0
        features["capital_run_length_longest"] = 0
        features["capital_run_length_total"] = 0

    return features


# --------------------------------------------------
# 3. Process all emails and run the classifier
# --------------------------------------------------

def parse_bodies(data, model):

    # Use the feature order recorded by the trained model
    # if it was trained using a Pandas DataFrame.
    if hasattr(model, "feature_names_in_"):
        feature_columns = list(model.feature_names_in_)
    else:
        # Fallback: the original Spambase feature order.
        feature_columns = (
            [f"word_freq_{word}" for word in SPAM_WORDS]
            + [f"char_freq_{char}" for char in SPECIAL_CHARACTERS]
            + [
                "capital_run_length_average",
                "capital_run_length_longest",
                "capital_run_length_total"
            ]
        )

    for email in data["emails"]:

        email_id = email["id"]
        body = email["body"]

        print(f"Processing email {email_id}...")

        # Calculate the features.
        features = parse_features(body)

        # Save calculated features into the email's JSON data.
        email["features"] = features

        # Convert the dictionary into a one-row DataFrame.
        # Explicitly arrange columns in the model's training order.
        feature_df = pd.DataFrame([features])
        feature_df = feature_df[feature_columns]

        # Run the prediction.
        prediction = int(model.predict(feature_df)[0])

        # Save the prediction: 1 = spam, 0 = not spam.
        email["label"] = prediction

        # Save the probability that the email is spam, if supported.
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(feature_df)[0]
            spam_class_index = list(model.classes_).index(1)

            email["spam_probability"] = float(
                probabilities[spam_class_index]
            )

        print(f"Email {email_id}: Prediction = {prediction}")

    return data


# --------------------------------------------------
# 4. Main program
# --------------------------------------------------

if __name__ == "__main__":

    # Load your saved model pipeline.
    model = joblib.load("spam_detector.joblib")

    # Load the email JSON file.
    with open("emails.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    # Extract features and classify every email.
    data = parse_bodies(data, model)

    # Save everything to a new JSON file.
    with open("emails_scored.json", "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    print("\nFinished processing all emails!")
    print("Results saved to emails_scored.json")
