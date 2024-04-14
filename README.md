# DLMI_course_project
Course project of Deep Learning in Medical Image.

# Project structure

- `output/`: The folder where the trained model, logs and plots are saved.
- `docs/`: The folder where contains the documentation of the project.
- `train.py`: Contains the training script(For model training only).
- `model.py`: Contains the model architecture.
- `evaluate.py`: Contains the evaluation script(You can train the model in this script if you want).
- `data_tool.py`: Contains functions for loading and splitting data.(Docs: [data_tool.md](docs/data_tool.md))

## Dataset structure

- Dataset_root
    - Train
        - Normal
        - Covid
    - Val
        - Normal
        - Covid
    - Test
        - Normal
        - Covid

## Dataset Citation
L. Wang, Z. Lin, A. Wong. "COVID-Net: a tailored deep convolutional neural network design for detection of COVID-19 cases from chest X-ray images," in Scientific Reports, vol. 10, no. 1, pp. 19549, 2020.

https://github.com/lindawangg/COVID-Net