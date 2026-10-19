#!/usr/bin/env python3
"""
Exercise 0: Environment and libraries verification
"""
import sys
import pandas as pd
import numpy as np
import matplotlib
import sklearn

def check_versions():
    print(f"Python version: {sys.version}")
    print(f"Pandas version: {pd.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Matplotlib version: {matplotlib.__version__}")
    print(f"Scikit-learn version: {sklearn.__version__}")

if __name__ == "__main__":
    check_versions()
    print("\nEnvironment setup verified successfully!")