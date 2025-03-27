## Workshop Overview 📋

This hands-on workshop will guide participants through the fundamentals of deep learning with PyTorch, covering:

- PyTorch's tensor operations and automatic differentiation
- Building and training neural networks for various tasks
- Implementing CNNs for computer vision and time series forecasting
- Applying transfer learning with pre-trained models
- Working with RNNs for sequential data
- Developing multi-class classification models

## Setting Up Your Development Environment 🛠️

### Recommended IDE: Visual Studio Code

We recommend using Visual Studio Code (VS Code) for this workshop as it provides excellent Python and Jupyter Notebook support.

#### Installing VS Code
1. Download VS Code from the [official website](https://code.visualstudio.com/).
2. Follow the installation instructions for your operating system.
3. Install the following extensions for a better experience:
   - Python
   - Jupyter
   - Pylance
   - Python Indent
   - IntelliCode

To install extensions in VS Code, click on the Extensions icon in the sidebar (or press `Ctrl+Shift+X`), search for each extension, and click "Install".

### Why Use a Virtual Environment? 🔒

Using a virtual environment is essential for Python development because it:

- Isolates project dependencies, preventing conflicts between projects
- Makes your project reproducible across different machines
- Allows easy management of package versions
- Keeps your global Python installation clean

### Setting Up a Virtual Environment 📦

#### Windows

```powershell
# Create a virtual environment
python -m venv dl_workshop_env

# Activate the environment
dl_workshop_env\Scripts\activate

# Deactivate the environment when finished
deactivate
```

#### Linux/macOS

```bash
# Create a virtual environment
python -m venv dl_workshop_env

# Activate the environment
source dl_workshop_env/bin/activate

# Deactivate the environment when finished
deactivate
```

### Required Libraries 📚

Install the necessary packages after activating your virtual environment:

```bash
pip install matplotlib numpy pandas scikit-learn jupyter urllib3 requests tqdm 
```

For GPU support (NVIDIA GPUs only):

```bash
# Install the latest version of PyTorch with CUDA support

pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# Note: Replace 'cu118' with the appropriate CUDA version for your system.
```

No GPU? No problem! PyTorch will automatically fall back to CPU. Install the CPU version of PyTorch using the command below:

```bash
pip install torch torchvision torchaudio
```

## Workshop Schedule ⏰

### Introduction (15 mins)
- Overview of PyTorch's tensor operations and automatic differentiation
- Building and training simple neural networks
- Creating training and evaluation pipelines
- Using GPUs for accelerated training

### The Mechanics of Deep Learning with PyTorch (120 mins)
- Train a basic PyTorch model to understand the deep learning workflow
- Implement CNNs to enhance accuracy in computer vision tasks
- Apply data augmentation techniques to improve model generalization
- Time Series Forecasting with CNNs

### Pre-trained Models and Recurrent Networks (120 mins)
- Use pre-trained models to solve deep learning challenges efficiently
- Apply transfer learning to fine-tune models for personalized applications
- Train RNNs on sequential data, including a text autocompletion task

### Final Project: Object Multi-class Classification (120 mins)
- Develop a classification model
- Implement data generators in PyTorch to handle small datasets efficiently
- Use transfer learning and feature extraction for faster model training

## Intended Learning Outcomes (ILOs) 🎯

By the end of the workshop, participants will be able to:
- Implement deep learning models using PyTorch for diverse applications
- Apply CNNs for both image classification and time series forecasting
- Utilize transfer learning to adapt pre-trained models to specific problems
- Train and optimize models for sequential data using RNNs
- Work with real-world datasets to solve classification and regression problems

## Getting Started 🚀

1. Clone this repository
2. Set up a virtual environment as described above
3. Install the required dependencies
4. Open the Jupyter notebooks in the repository to follow along with the workshop materials

## Support 💬

For issues or questions during the workshop, please reach out to the workshop instructors.

## Prerequisites 📝

- Basic Python programming knowledge
- Familiarity with fundamental machine learning concepts
- A laptop with Python 3.8+ installed