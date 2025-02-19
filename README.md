# Neural Network for Multi-Class Classification

This repository contains the implementation of a neural network designed for multi-class classification, developed as part of the "Web and Information Retrieval" course at the Università degli Studi del Sannio.

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Usage](#usage)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Run the Jupyter Notebook (Optional)](#2-run-the-jupyter-notebook-optional)
  - [3. Execute the Main Script](#3-execute-the-main-script)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Project Overview

The project focuses on building a neural network with a softmax activation function in the final layer, enabling it to handle multi-class classification tasks effectively. The implementation includes data preprocessing, model training, evaluation, and visualization of results.

## Repository Structure

```
Web-Inf/
├── main.py                  # Main script for model training and evaluation
├── tesina_finale_softmax.ipynb  # Detailed notebook explaining the project workflow
├── resources/              # Directory containing datasets and additional resources
├── results/                # Directory to store training results and figures
├── .gitignore              # Files and folders to be ignored by git
├── LICENSE                 # Project license information
└── README.md               # Project documentation
```

## Requirements

To run the code in this repository, ensure you have the following dependencies installed:

- Python 3.x
- TensorFlow
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Jupyter Notebook (optional, for exploring the notebook)

You can install the required packages using pip:

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib jupyter
```

## Usage

### 1. Clone the Repository

```bash
git clone https://github.com/sroman0/Web-Inf.git
cd Web-Inf
```

### 2. Run the Jupyter Notebook (Optional)

To explore the project workflow in an interactive environment:

```bash
jupyter notebook tesina_finale_softmax.ipynb
```

### 3. Execute the Main Script

To train and evaluate the model, run:

```bash
python main.py
```

Ensure that the dataset is correctly referenced within the script and that the `resources/` directory contains the necessary data files.

## Model Architecture

The implemented neural network consists of:

1. **Input Layer**: Processes preprocessed feature vectors.
2. **Hidden Layers**: Multiple dense layers with ReLU activation functions to capture complex patterns.
3. **Output Layer**: Softmax activation function for multi-class classification.

The network is optimized using the Adam optimizer and uses categorical cross-entropy as the loss function. Model performance is tracked using accuracy metrics.

## Results

- **Accuracy & Loss Plots**: Training history is visualized to assess model convergence.
- **Confusion Matrix**: Provides insights into class-wise prediction accuracy.
- **Evaluation Metrics**: Detailed performance metrics, including precision, recall, and F1-score.
- All generated results and plots are saved in the `results/` directory.

## License

This project is licensed under the **GNU General Public License v3.0**.  
See the [LICENSE](https://github.com/sroman0/Web-Inf/blob/main/LICENSE) file for more details.

## Acknowledgments

This project was developed as part of the *"Web and Information Retrieval"* course at **Università degli Studi del Sannio**. Special thanks to the course instructors and fellow students for their valuable guidance and support.

---

For questions or further information, please contact the repository maintainer or refer to the course materials provided by the university.
