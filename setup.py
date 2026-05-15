from setuptools import setup, find_packages

setup(
    name="scvla",
    version="0.1.0",
    description="Self-Correcting VLA: lightweight correction module for any VLA",
    author="lexus-x",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "numpy>=1.24",
    ],
    extras_require={
        "dev": ["pytest", "scikit-learn"],
        "libero": ["libero"],
        "metaworld": ["metaworld"],
    },
)
