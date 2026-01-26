"""
Helper functions for feature generation and metal counting.

This module provides utility functions for:
- Counting metal atoms in different zones
- Calculating multiplicities for surface configurations
- Removing zero columns from feature matrices
- Sorting metal lists
"""

import numpy as np
from math import factorial
import itertools as it


def count_metals(metals, nMetals):
    """
    Count occurrences of each metal type in the ensemble.
    
    Args:
        metals (list of int): List of metal IDs (0, 1, 2 for Ni, Fe, Co)
        nMetals (int): Total number of metal types (e.g., 3 for ternary)
    
    Returns:
        list of int: Count of each metal type
        
    Example:
        >>> count_metals([0, 0, 1], 3)
        [2, 1, 0]  # 2 Ni, 1 Fe, 0 Co
    """
    counts = [0] * nMetals
    for metal in metals:
        counts[metal] += 1
    return counts


def count_atoms(userSymbols, refSymbols):
    """
    Count atoms by symbol, ordered according to reference list.
    
    Args:
        userSymbols (list or str): Chemical symbols to count
        refSymbols (list of str): Reference order for counting (e.g., ['Ni', 'Fe', 'Co'])
    
    Returns:
        list of int: Count of each symbol in refSymbols order
        
    Example:
        >>> count_atoms(['Ni', 'Ni', 'Fe'], ['Ni', 'Fe', 'Co'])
        [2, 1, 0]
    """
    nRef = len(refSymbols)
    counts = [0] * nRef
    
    if isinstance(userSymbols, list):
        if not userSymbols:
            return []
        for symbol in userSymbols:
            for i, refSymbol in enumerate(refSymbols):
                if symbol == refSymbol:
                    counts[i] += 1
    else:  # userSymbols is a string
        for i, refSymbol in enumerate(refSymbols):
            if userSymbols == refSymbol:
                counts[i] += 1
    return counts


def sortMetals(metalList, metals):
    """
    Sort metal list according to a reference order.
    
    Args:
        metalList (list of str): Metals to sort (e.g., ['Fe', 'Ni', 'Co'])
        metals (list of str): Reference order (e.g., ['Ni', 'Fe', 'Co'])
    
    Returns:
        list of str: Sorted metal list
        
    Example:
        >>> sortMetals(['Fe', 'Ni', 'Co'], ['Ni', 'Fe', 'Co'])
        ['Ni', 'Fe', 'Co']
    """
    nMetals = len(metals)
    n = len(metalList)
    metalNum = [0] * n
    
    # Assign numbers to metals
    for i in range(n):
        for j in range(nMetals):
            if metalList[i] == metals[j]:
                metalNum[i] = j
    
    # Sort by assigned numbers
    metalNum = sorted(metalNum)
    
    # Convert back to symbols
    sortedMetals = [''] * n
    for i in range(n):
        for j in range(nMetals):
            if metalNum[i] == j:
                sortedMetals[i] = metals[j]
    
    return sortedMetals


def remove_zero_columns(X):
    """
    Remove columns that are all zeros from feature matrix.
    
    Args:
        X (np.ndarray): Feature matrix (samples × features)
    
    Returns:
        tuple: (X_reduced, removeIds, keepIds)
            - X_reduced: Matrix with zero columns removed
            - removeIds: Indices of removed columns
            - keepIds: Indices of kept columns
    """
    keepIds, removeIds = [], []
    for i, column in enumerate(X.T):
        if np.array_equal(column, np.zeros(len(X))):
            removeIds.append(i)
        else:
            keepIds.append(i)
    return X[:, keepIds], removeIds, keepIds


def unique(nMetals, nSites):
    """
    Calculate number of unique configurations.
    
    Formula: (nMetals + nSites - 1)! / (nSites! × (nMetals - 1)!)
    This is "stars and bars" combinatorics problem.
    
    Args:
        nMetals (int): Number of metal types
        nSites (int): Number of sites
    
    Returns:
        float: Number of unique configurations
    """
    return (factorial(nMetals + nSites - 1) / 
            (factorial(nSites) * factorial(nMetals - 1)))


def multiplicity(nAtoms, nEachMetal):
    """
    Calculate multiplicity (number of equivalent arrangements).
    
    Multiplicity = nAtoms! / (n1! × n2! × ... × nM!)
    where ni is the count of metal i.
    
    Args:
        nAtoms (int): Total number of atoms in zone
        nEachMetal (list of int): Count of each metal type
    
    Returns:
        float: Multiplicity factor
        
    Example:
        >>> multiplicity(3, [2, 1, 0])
        3.0  # 3!/(2!×1!×0!) = 6/(2×1×1) = 3
    """
    product = 1
    for nMetal in nEachMetal:
        product *= factorial(nMetal)
    return factorial(nAtoms) / product
