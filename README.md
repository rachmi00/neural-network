# Neural Network Implementation for MNIST Dataset

## Project Overview
This project implements a simple neural network from scratch using Python to classify handwritten digits from the MNIST dataset. The implementation is contained in `neural_network.ipynb` Jupyter notebook.

## Dependencies
- Python 3.x
- NumPy (2.0 or later)
- SciPy
- Matplotlib
- Jupyter Notebook

## Installation
1. Create a virtual environment (recommended):
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows, use: myenv\Scripts\activate
```

2. Install required packages:
```bash
pip install numpy scipy matplotlib jupyter
```

3. Download the MNIST dataset CSV files and place them in a folder named `mnist_dataset` in your project directory:
   - mnist_train.csv
   - mnist_test.csv
   Download from source : http://www.pjreddie.com/media/files/mnist_train.csv
                          http://www.pjreddie.com/media/files/mnist_test.csv

## Neural Network Architecture
- Input Layer: 784 nodes (28x28 pixels)
- Hidden Layer: 100 nodes
- Output Layer: 10 nodes (one for each digit 0-9)
- Activation Function: Sigmoid (implemented using scipy.special.expit)
- Learning Rate: 0.1

## Implementation Details
The neural network implementation includes:
- Weight initialization using normal distribution
- Forward propagation through hidden and output layers
- Backpropagation for weight updates
- Training loop with multiple epochs
- Testing functionality with accuracy calculation

## Usage
1. Open `neural_network.ipynb` in Jupyter Notebook:
```bash
jupyter notebook
```

2. Run all cells in the notebook to:
   - Initialize the neural network
   - Train the model on the MNIST training dataset
   - Test the model's performance on the test dataset

## Data Preprocessing
- Input data is scaled from 0-255 to 0.01-0.99 range
- Target values are set to 0.99 for correct digit and 0.01 for others
- Training data is processed in multiple epochs

## Performance
The network's performance is calculated as the ratio of correct predictions to total number of test cases. Typical performance varies based on:
- Number of epochs
- Learning rate
- Hidden layer size
- Weight initialization

## File Structure
```
project_root/
│
├── neural_network.ipynb
├── README.md
│
└── mnist_dataset/
    ├── mnist_train.csv
    └── mnist_test.csv
```

## Limitations and Future Improvements
- Single hidden layer architecture
- Fixed learning rate
- No regularization
- Basic preprocessing
- No validation set

Potential improvements could include:
- Adding multiple hidden layers
- Implementing different activation functions
- Adding dropout for regularization
- Including batch processing
- Adding learning rate decay
- Implementing cross-validation

## Contributing
Feel free to fork this repository and submit pull requests for improvements. Some areas that could benefit from contributions:
- Code optimization
- Additional features
- Better documentation
- Performance improvements
- Testing suite

## License
This project is open source and available under the MIT License.

## Acknowledgments
This implementation is based on basic neural network principles and is designed for educational purposes to understand the fundamentals of neural networks and backpropagation.