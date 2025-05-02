![](https://i.imgur.com/mTltzAc.png)
## Workshop Overview 📋

This hands-on workshop will guide participants through the fundamentals of deep learning with PyTorch, covering:

- PyTorch's tensor operations and automatic differentiation
- Building and training neural networks for various tasks
- Implementing CNNs for computer vision
- Applying transfer learning with pre-trained models
- Working with sequential data
- Developing multi-class classification models

## Setting Up Your Development Environment 🛠️

### Recommended Platform: Google Colab

We recommend using [Google Colab](https://colab.research.google.com/) for this workshop as it provides a free cloud-based environment with GPU support for running Jupyter notebooks.

### Prerequisites for Google Colab

- A Google account is required to access Google Colab.
- Ensure you have a stable internet connection.

### Running the Notebooks on Google Colab

1. Open the workshop repository on GitHub.
2. Click on the "Open in Colab" badge [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() (if available) **or** manually upload the notebook:
   - Download the notebook to your local machine.
   - Go to [Google Colab](https://colab.research.google.com/).
   - Click on "File > Upload Notebook" and select the downloaded notebook.

### Setting Up the GPU Environment on Colab

1. Open the notebook in Google Colab.
2. Navigate to "Runtime > Change runtime type".
3. In the "Hardware accelerator" dropdown, select "GPU".
4. Click "Save" to apply the changes.

### Getting familiar with Google Colab

- [Google Colab Tips and Tricks](https://colab.research.google.com/notebooks/basic_features_overview.ipynb) - Overview of basic features in Google Colab
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html) - Frequently asked questions about Google Colab

⚠️ **Note**: Google Colab has a maximum session time limit of 12 hours. If your session times out, you may need to re-run the notebook from the beginning. Save your work frequently to avoid data loss.

### Installing Required Libraries in Colab

At the beginning of each notebook, ensure you run the firtst code cell to install the required libraries.

For GPU support, PyTorch will automatically utilize the GPU if available.

## Workshop Content 📖

### Session 1: Introduction to PyTorch (~0.5 hours)
- PyTorch basics and tensors
- Creating and manipulating tensors
- Indexing and slicing tensors
- Tensor operations (arithmetic, matrix operations)
- Broadcasting
- Reshaping tensors
- Automatic differentiation with autograd
- Working with data in PyTorch
- Using GPUs for acceleration

### Session 2: Artificial Neural Networks (~1 hour)
- Understanding neurons and perceptrons
- Forward and backward pass
- Neural network components
- Case study: Space Shuttle Challenger disaster
- Implementing a simple neural network from scratch
- Training neural networks
- Evaluation and prediction

### Session 3: Training Neural Networks (~1.5 hours)
- PyTorch workflow for training models
- Working with the ARKOMA robotics dataset / PDE with Physics Informed Neural Networks
- Data preparation and normalization
- Activation functions and their purposes
- Model architecture design
- Weight initialization techniques
- Choosing optimizers and loss functions
- Creating effective training loops
- Handling overfitting and early stopping
- Model evaluation techniques

### Session 4: Convolutional Neural Networks (~1.5 hours)
- CNN fundamentals and operations
- Convolution basics and filter design
- Implementing custom filters
- Image data preparation with transforms
- Building CNN models from scratch
- Historical Crack Dataset exploration
- Using pooling layers
- Regularization techniques
- CNN architectures (SimpleNet and TinyVGG)
- Training and evaluating CNN models

### Session 5: Transfer Learning (~1.5 hours)
- Transfer learning fundamentals
- Medical image segmentation with ISIC dataset
- Data preparation for segmentation tasks
- U-Net architecture implementation
- Segmentation-specific loss functions (Dice loss)
- Using pre-trained models as feature extractors
- EfficientNet-based U-Net implementation
- Comparing models trained from scratch vs. transfer learning

## Intended Learning Outcomes (ILOs) 🎯

By the end of the workshop, participants will be able to:
- Implement deep learning models using PyTorch for diverse applications
- Apply CNNs for image classification and segmentation tasks
- Utilize transfer learning to adapt pre-trained models to specific problems
- Work with real-world datasets to solve classification and regression problems

## Workshop Materials Structure 📁

### Repository Organization
```
UoM_fse_dl_workshop/
├── datasets/                  # Dataset storage
├── solutions/                 # Solved notebooks
├── figs/                      # Workshop images and icons
├── utils/                     # Utility functions and checkers
│   ├── core.py                # Exercise checker implementation
│   ├── data.py                # Data loading utilities
│   ├── ml/                    # ML helper functions
│   └── solutions.json         # Exercise solutions and hints
└── SE01_CA_Intro_to_pytorch.ipynb    # Workshop notebooks
```

### Using the Exercise Checker 📝

Each workshop notebook contains exercises marked with 🎯. The exercise checker helps you verify your solutions and provides hints when needed.

#### Checking Your Solutions
```python
# At the end of each exercise, there will be a check block:
answer = {
    'your_solution': your_tensor,
    'another_part': another_result
}
checker.check_exercise(exercise_number, answer)
```

#### Getting Hints
If you're stuck, you can get hints for any exercise:
```python
# Display hints for exercise 1
checker.display_hints(1)
```

#### Exercise Feedback
The checker provides immediate feedback:
- ✅ Correct solutions are marked with a green checkmark
- ❌ Incorrect solutions show what went wrong
- 💡 Helpful hints appear when needed

### Common Workflows

1. **Starting an Exercise**:
   - Read the exercise description carefully
   - Implement your solution in the provided code cell
   - Run the cell to see if it works

2. **Checking Your Work**:
   - The checker automatically validates your solution
   - Multiple aspects may be checked (values, shapes, types)
   - All parts must be correct to pass

3. **Getting Help**:
   - Use `checker.display_hints()` for guidance
   - Hints are context-aware and specific to each exercise
   - Multiple hints may be available per exercise

4. **Learning from Mistakes**:
   - Pay attention to error messages
   - Check tensor shapes and types when debugging
   - Use print statements to understand intermediate results

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
- Basic understanding of linear algebra and calculus (optional but helpful)
- Familiarity with Jupyter Notebooks (optional but helpful)

## Additional Resources 📚

To further enhance your learning experience, here are some valuable external resources:

### PyTorch Documentation

- [PyTorch Official Documentation](https://pytorch.org/docs/stable/index.html) - Comprehensive API reference and tutorials
- [PyTorch Cheat Sheet](https://pytorch.org/tutorials/beginner/ptcheat.html) - Quick reference guide for common PyTorch operations
- [PyTorch Examples Repository](https://github.com/pytorch/examples) - Official examples for various deep learning tasks

### Pre-trained Models & Transfer Learning

- [torchvision.models](https://pytorch.org/vision/stable/models.html) - Documentation for pre-trained models in torchvision
- [PyTorch Hub](https://pytorch.org/hub/) - Repository of pre-trained models ready for fine-tuning
- [Hugging Face Models](https://huggingface.co/models) - Thousands of pre-trained models for various tasks

### Visual Learning Tools

- [CNN Explainer](https://poloclub.github.io/cnn-explainer/) - Interactive visualization of convolutional neural networks
- [Distill: Feature Visualization](https://distill.pub/2017/feature-visualization/) - Understanding neural networks through feature visualization
- [TensorBoard](https://www.tensorflow.org/tensorboard) - Visualization toolkit compatible with PyTorch (via torch.utils.tensorboard)

### Tutorials & Courses

- [Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) - Quick PyTorch introductory tutorial
- [PyTorch Tutorials](https://pytorch.org/tutorials/) - Official collection of tutorials from basic to advanced topics
- [Fast.ai Practical Deep Learning](https://course.fast.ai/) - Practical deep learning course using PyTorch

### Papers & Research

- [U-Net Paper](https://arxiv.org/abs/1505.04597) - Original U-Net architecture paper for biomedical image segmentation
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946) - Scaling up CNNs more efficiently
- [Transfer Learning Survey](https://arxiv.org/abs/1911.02685) - Comprehensive survey on transfer learning

### Books

- [Dive into Deep Learning](https://d2l.ai/) - Interactive book with code examples in multiple frameworks including PyTorch

## VS Code Setup Guide for Workshop Participants

**Note**: The workshop now uses Google Colab as the primary platform. If you prefer using VS Code, refer to the archived instructions in the repository.
