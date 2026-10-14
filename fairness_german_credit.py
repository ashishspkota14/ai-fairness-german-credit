"""
Detecting and Mitigating Age Bias in Credit Decisions with AI Fairness 360
-------------------------------------------------------------------------
This script uses IBM's open-source AI Fairness 360 (AIF360) toolkit on the
German Credit dataset (1,000 loan applicants labeled as good or bad credit risk).

Protected attribute: age
    Privileged group:   applicants aged 25 or older
    Unprivileged group: applicants younger than 25

Steps:
    1. Measure bias in the original training data.
    2. Mitigate the bias with the Reweighing pre-processing algorithm, which
       gives each training example a weight so that favorable outcomes are
       balanced across age groups, without changing any applicant's data.
    3. Train a logistic regression loan-approval model on the original data and
       on the reweighed data, then compare accuracy and fairness on a test set.

Usage:
    python fairness_german_credit.py

Author: Aashish Sapkota
Course: MSCS-633-A01 Advanced Artificial Intelligence
"""

import logging
import os
import urllib.request

# AIF360 logs warnings about optional add-ons (TensorFlow, etc.) that this
# script does not use; hide them so the output stays readable.
logging.basicConfig(level=logging.ERROR)

import matplotlib

matplotlib.use("Agg")  # save charts to files without opening a window
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import aif360
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import GermanDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

RANDOM_SEED = 1  # fixed seed so the split and results are repeatable
OUTPUT_DIR = "outputs"

# Group definitions: 1 = aged 25 or older (privileged), 0 = younger than 25
PRIVILEGED = [{"age": 1}]
UNPRIVILEGED = [{"age": 0}]

UCI_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/"


def ensure_german_data():
    """AIF360 does not ship the raw data, so download it into AIF360's data folder if missing."""
    folder = os.path.join(os.path.dirname(aif360.__file__), "data", "raw", "german")
    path = os.path.join(folder, "german.data")
    if not os.path.exists(path):
        print("Downloading german.data from the UCI Machine Learning Repository...")
        urllib.request.urlretrieve(UCI_BASE + "german.data", path)


def load_dataset():
    """Load German Credit with age as the protected attribute (as in IBM's AIF360 tutorial)."""
    return GermanDataset(
        protected_attribute_names=["age"],
        privileged_classes=[lambda age: age >= 25],
        features_to_drop=["personal_status", "sex"],  # keep the study focused on age
    )


def dataset_bias(dataset):
    """Return bias metrics for a labeled dataset (before any model is trained)."""
    m = BinaryLabelDatasetMetric(dataset, unprivileged_groups=UNPRIVILEGED, privileged_groups=PRIVILEGED)
    return {
        "Favorable rate, age 25+": m.base_rate(privileged=True),
        "Favorable rate, under 25": m.base_rate(privileged=False),
        "Statistical parity difference": m.statistical_parity_difference(),
        "Disparate impact": m.disparate_impact(),
    }


def train_and_evaluate(train, test, use_weights):
    """Train a logistic regression model and measure its accuracy and fairness on the test set."""
    scaler = StandardScaler().fit(train.features)
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    weights = train.instance_weights if use_weights else None
    model.fit(scaler.transform(train.features), train.labels.ravel(), sample_weight=weights)

    # Copy the test set and replace its labels with the model's predictions
    predictions = test.copy(deepcopy=True)
    predictions.labels = model.predict(scaler.transform(test.features)).reshape(-1, 1)

    m = ClassificationMetric(test, predictions, unprivileged_groups=UNPRIVILEGED, privileged_groups=PRIVILEGED)
    balanced_accuracy = 0.5 * (m.true_positive_rate() + m.true_negative_rate())
    return {
        "Accuracy": m.accuracy(),
        "Balanced accuracy": balanced_accuracy,
        "Approval rate, age 25+": m.selection_rate(privileged=True),
        "Approval rate, under 25": m.selection_rate(privileged=False),
        "Statistical parity difference": m.statistical_parity_difference(),
        "Disparate impact": m.disparate_impact(),
        "Equal opportunity difference": m.equal_opportunity_difference(),
        "Average odds difference": m.average_odds_difference(),
    }


def print_comparison(title, original, mitigated):
    """Print original and mitigated metrics side by side."""
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)
    print(f"{'Metric':<34}{'Original':>12}{'Mitigated':>14}")
    print("-" * 72)
    for name in original:
        print(f"{name:<34}{original[name]:>12.4f}{mitigated[name]:>14.4f}")


def save_chart(data_orig, data_mitig, model_orig, model_mitig):
    """Save a bar chart comparing fairness metrics before and after mitigation."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    labels = ["Data: statistical\nparity difference", "Model: statistical\nparity difference",
              "Model: equal\nopportunity difference", "Model: average\nodds difference"]
    keys = [(data_orig, data_mitig, "Statistical parity difference"),
            (model_orig, model_mitig, "Statistical parity difference"),
            (model_orig, model_mitig, "Equal opportunity difference"),
            (model_orig, model_mitig, "Average odds difference")]
    before = [o[k] for o, _, k in keys]
    after = [m[k] for _, m, k in keys]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars_b = ax.bar(x - 0.2, before, 0.4, label="Original", color="#dc2626")
    bars_a = ax.bar(x + 0.2, after, 0.4, label="Mitigated (Reweighing)", color="#2563eb")
    ax.bar_label(bars_b, fmt="%.3f", padding=3)
    ax.bar_label(bars_a, fmt="%.3f", padding=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhspan(-0.1, 0.1, color="#16a34a", alpha=0.08, label="Fair range (-0.1 to 0.1)")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Difference (unprivileged - privileged)")
    ax.set_title("Age bias in German Credit decisions: original vs. mitigated (0 = fair)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fairness_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nChart saved to {path}")


def main():
    np.random.seed(RANDOM_SEED)
    ensure_german_data()

    dataset = load_dataset()
    train, test = dataset.split([0.7], shuffle=True, seed=RANDOM_SEED)
    print(f"German Credit dataset: {dataset.features.shape[0]} applicants, "
          f"{dataset.features.shape[1]} features")
    print(f"Training set: {train.features.shape[0]}   Test set: {test.features.shape[0]}")
    print("Protected attribute: age (privileged = 25 or older, unprivileged = under 25)")

    # Step 1 and 2: dataset bias before and after Reweighing
    reweigher = Reweighing(unprivileged_groups=UNPRIVILEGED, privileged_groups=PRIVILEGED)
    train_reweighed = reweigher.fit_transform(train)
    data_orig = dataset_bias(train)
    data_mitig = dataset_bias(train_reweighed)
    print_comparison("STEP 1 - BIAS IN THE TRAINING DATA (original vs. reweighed)", data_orig, data_mitig)

    # Step 3: loan-approval model trained without and with the Reweighing weights
    model_orig = train_and_evaluate(train, test, use_weights=False)
    model_mitig = train_and_evaluate(train_reweighed, test, use_weights=True)
    print_comparison("STEP 2 - LOAN APPROVAL MODEL ON THE TEST SET (original vs. mitigated)", model_orig, model_mitig)

    print("\nFair values: differences close to 0 and disparate impact close to 1 "
          "(the common 80% rule accepts 0.8 to 1.25).")
    save_chart(data_orig, data_mitig, model_orig, model_mitig)


if __name__ == "__main__":
    main()
