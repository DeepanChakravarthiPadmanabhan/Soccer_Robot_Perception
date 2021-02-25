import os
from setuptools import find_packages, setup


__version__ = "0.1"

if "VERSION" in os.environ:
    BUILD_NUMBER = os.environ["VERSION"].rsplit(".", 1)[-1]
else:
    BUILD_NUMBER = os.environ.get("BUILD_NUMBER", "dev")

dependencies = [
    "click>=7.0",
    "numpy>=1.17.0,<2.0",
    "torch",
    "torchvision",
    "opencv-python>=4.1.1.26,<5.0",
    "tensorboard==2.3.0",
    "scikit-learn>=0.21.0,<1.0",
    "scikit-image",
    "git-python>=1.0.3",
    "gin-config==0.3.0",
    "torchsummary>=1.5.1",
    "xlsxwriter>=1.2.9",
    "black",
]

setup(
    name="soccer_robot_perception",
    version="{0}.{1}".format(__version__, BUILD_NUMBER),
    description="Soccer Robot Perception -- A perception pipeline for robots playing soccer",
    author="Deepan Chakravarthi Padmanabhan, Mihir Mulye",
    install_requires=dependencies,
    packages=find_packages(),
    zip_safe=False,
    entry_points=dict(
        console_scripts=[
            "soccer_robot_perception_train=soccer_robot_perception.train:train",
            "soccer_robot_perception_evaluate=soccer_robot_perception.evaluate.evaluate:evaluate",
        ]
    ),
    data_files=[
        (
            "soccer_robot_perception_config",
            [
                "config/train.gin",
                "config/evaluate.gin",
            ],
        )
    ],
    python_requires=">=3.7,<3.9",
)
