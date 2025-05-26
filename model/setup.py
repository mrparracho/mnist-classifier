from setuptools import setup, find_packages

setup(
    name="mnist-classifier",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "torch",
        "torchvision",
        "pillow",
        "psycopg2-binary",
        "python-multipart",
    ],
) 