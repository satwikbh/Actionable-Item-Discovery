# Actionable Item Discovery

This project implements a system to identify "actionable items" from text, primarily focusing on email subjects and messages. It combines linguistic heuristic rules with various machine learning models to classify sentences as actionable or not.

## Features

*   **Data Normalization:** Cleans and preprocesses input text by removing invalid characters, forwarded messages, and common stopwords.
*   **Heuristic Rule Engine:** Applies linguistic rules, including Named Entity Recognition (NER) for time and Part-of-Speech (POS) tagging, to identify potential actionable content.
*   **Machine Learning Classification:** Utilizes a suite of machine learning models to classify sentences based on features derived from the heuristic rules and pre-tagged data.
*   **Model Evaluation:** Reports performance metrics (accuracy, precision, recall, F1-score) for the trained models.
*   **Interactive Prediction:** Allows users to input sentences and receive predictions from both the heuristic and the best-performing ML model.

## Installation

To set up the project, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Actionable-Item-Discovery.git
    cd Actionable-Item-Discovery
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download spaCy language model:**
    The `run.sh` script handles this, or you can run it manually:
    ```bash
    python -m spacy download en_core_web_sm
    ```

## Usage

The `TestModel.py` script is the main entry point for training models and making predictions.

### Training Models

To train and generate all the machine learning models, run `TestModel.py` with the `test` flag set to `False` (this is the default behavior if no arguments are provided or if `test=False` is explicitly set in the script):

```bash
python TestModel.py
```

### Making Predictions

To use the trained models for interactive predictions, run `TestModel.py` with the `test` flag set to `True`:

```bash
# You might need to modify TestModel.py to set the test flag to True
# or pass it as a command-line argument if the script supports it.
# Example (assuming TestModel.py accepts a --test argument):
python TestModel.py --test
```

Upon execution, you will be prompted to enter a sentence. The system will then provide two predictions:

1.  **Heuristic Model Prediction:** Based on linguistic rules.
2.  **ML Model Prediction:** Based on the best-performing machine learning model (MLP).

The final prediction is an aggregation of both outputs.

## Project Structure

*   `Core/`: Contains core logic for data processing, heuristic rules, and data reading.
    *   `HeuristicRules.py`: Implements the linguistic rules.
    *   `Logic.py`: Core classification logic.
    *   `ProcessData.py`: Handles data normalization and preparation.
    *   `ReadData.py`: Manages data input.
*   `Models/`: Houses the implementations of various machine learning models.
    *   `Adaboost.py`
    *   `DecisionTree.py`
    *   `ExtraTree.py`
    *   `LRModel.py` (Logistic Regression)
    *   `MLP.py` (Multi-Layer Perceptron)
    *   `NaiveBayes.py`
    *   `RandomForest.py`
    *   `Metrics.py`: Contains functions for evaluating model performance.
*   `Utils/`: Utility functions and configurations.
    *   `ConfigUtil.py`: Utility for reading configuration.
    *   `Helper.py`, `HelperClass.py`: General helper functions.
    *   `LoggerUtil.py`, `logging.json`: Logging utilities.
*   `CheckPreTaggedData.py`: Script for checking pre-tagged data.
*   `Config.json`: Configuration file for data paths and parameters.
*   `LinguisticModel.py`: Likely integrates linguistic processing.
*   `requirements.txt`: Lists Python dependencies.
*   `run.sh`: Setup script for virtual environment and spaCy model download.
*   `TestModel.py`: Main script for model training and prediction.
*   `Images/`: Stores project-related images, e.g., confusion matrices.

## Models Used

The project utilizes the following machine learning models for classification:

*   Naive Bayes
*   Logistic Regression
*   Decision Tree
*   Random Forest
*   Extra Tree
*   Adaboost
*   Multi-Layer Perceptron (MLP)

## Results

The Multi-Layer Perceptron (MLP) model achieved the best performance with an accuracy of 95%. The precision, recall, and F1-score values were all reported as 0.94.

When run on pre-tagged texts, the model reportedly achieved 100% accuracy, suggesting that noise in the real-world data contributes to the decrease in accuracy.

![Confusion Matrix](Images/confusion_matrix_Adaboost_test.png)
