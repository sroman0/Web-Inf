# Neural Network for Multi-Class Classification

![Università degli Studi del Sannio Logo](resources/Unisannio_logo.png))

This repository contains the implementation of a neural network designed for multi-class classification, developed as part of the **"Web and Information Retrieval"** course at the **Università degli Studi del Sannio**.

## Project Overview

The project focuses on building a neural network with a **softmax activation function** in the final layer, enabling it to handle **multi-class classification tasks** effectively. The implementation includes data preprocessing, model training, evaluation, and visualization of results.

## Repository Structure

- **`main.py`**: The main script that initializes and trains the neural network model.
- **`tesina_finale_softmax.ipynb`**: A Jupyter Notebook providing a detailed walkthrough of the project, including data preprocessing, model architecture, training process, and evaluation metrics.
- **`dataset/`**: Directory containing the dataset used for training and testing.
- **`models/`**: Directory for saving trained models.
- **`results/`**: Contains plots and logs of training performance.
- **`requirements.txt`**: List of dependencies required to run the project.
- **`LICENSE`**: The project's license information.
- **`README.md`**: This file, offering an overview of the project.

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
pip install -r requirements.txt
```

## Usage

### 1. Clone the Repository

```bash
git clone https://github.com/sroman0/Web-Inf.git
cd Web-Inf
```

### 2. Run the Jupyter Notebook (Optional)

If you want to explore the Jupyter Notebook:

```bash
jupyter notebook tesina_finale_softmax.ipynb
```

### 3. Execute the Main Script

To train and evaluate the model, run:

```bash
python main.py
```

Ensure that the dataset is properly referenced within the script.

## Model Architecture

The implemented neural network consists of multiple layers:

1. **Input Layer**: Accepts preprocessed feature vectors.
2. **Hidden Layers**: Multiple dense layers with activation functions (ReLU) to capture complex patterns.
3. **Output Layer**: Softmax activation function for multi-class classification.

The network is trained using **categorical cross-entropy loss** and optimized with **Adam optimizer**. The model's performance is evaluated using accuracy metrics and visualized through learning curves.

## Results

- Model performance is evaluated based on **accuracy, loss, and confusion matrix**.
- Training history is visualized using **Matplotlib** to track accuracy and loss over epochs.
- Results are stored in the `results/` directory.

## License

This project is licensed under the **GPL-3.0 License**. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

This project was developed as part of the **Web and Information Retrieval** course at **Università degli Studi del Sannio**. Special thanks to the course instructors and colleagues for valuable insights.

---

For any questions or further information, please contact the repository owner or refer to the course materials provided by the Università degli Studi del Sannio.
