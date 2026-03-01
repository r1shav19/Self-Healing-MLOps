from setuptools import setup, find_packages

setup(
    name="self-healing-mlops",
    version="0.1.0",
    description="Self-healing MLOps platform with automated retraining and monitoring",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        line.strip()
        for line in open("requirements.txt").readlines()
        if line.strip() and not line.startswith("#")
    ],
    extras_require={
        "dev": [
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "isort>=5.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mlops-train=src.train:main",
        ],
    },
)
