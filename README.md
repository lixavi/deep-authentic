## Deep Authentic

DeepAuthentic is a tool for DeepFake analysis and creation, using Generative Adversarial Networks (GANs) for media synthesis and detection.

## Purpose

The purpose of DeepAuthentic is to provide users with a comprehensive tool for analyzing and creating DeepFake content. It includes functionalities for both detecting DeepFake content in images and videos, as well as generating DeepFake content using pre-trained GAN models.

## Usage

### GUI Mode

To run the tool in GUI mode, execute the following command:

```python main.py```

This will launch the graphical user interface where you can perform DeepFake analysis and creation using the provided buttons.

### CLI Mode

To run the tool in CLI mode, you can use the individual Python scripts provided in the project. For example:

```
python detection.py --image path/to/image.jpg
python synthesis.py --face path/to/face_image.jpg
```

Replace `path/to/image.jpg` and `path/to/face_image.jpg` with the paths to the input images.
