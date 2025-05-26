from setuptools import setup, find_packages

setup(
    name="mnist-classifier",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "sqlalchemy",
        "pytest",
        "torch",
        "numpy",
        "streamlit",
        "python-multipart",
    ],
    python_requires=">=3.8",
) 