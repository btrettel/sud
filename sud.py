#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import numpy as np
import matplotlib.pyplot as plt
import pint
from uncertainties import ufloat, unumpy
import sys
import warnings
import scipy.stats

interval_probability_level = 0.95

def has_uncertainty(x):
    # This will work for floats, arrays, and ndarrays.
    
    try:
        # assume that this is an array or ndarray at first
        len(x) # This will fail if not an array or ndarray.
        
        for val in x:
            try:
                val.std_dev
            except:
                return False
        
        return True
    except:
        try:
            x.std_dev
            return True
        except:
            return False

def test_has_uncertainty():
    # float without uncertainty
    x = 0.20
    assert(not(has_uncertainty(x)))
    
    # float with uncertainty
    x = ufloat(0.20, 0.01)
    assert(has_uncertainty(x))
    
    # ndarray without uncertainty
    x = np.array([1., 2.])
    assert(not(has_uncertainty(x)))
    
    # ndarray with uncertainty
    x = unumpy.uarray([1, 2], [0.01, 0.002])
    assert(has_uncertainty(x))
    
    # float with units without uncertainty
    x = 0.20 * ureg.meter
    assert(not(has_uncertainty(x)))
    
    # float with units with uncertainty
    x = ufloat(0.20, 0.01) * ureg.meter
    assert(has_uncertainty(x))
    
    # ndarray with units without uncertainty
    x = ureg.Quantity(np.array([1., 2.]), ureg.meter)
    assert(not(has_uncertainty(x)))
    
    # ndarray with units with uncertainty
    x = ureg.Quantity(unumpy.uarray([1, 2], [0.01, 0.002]), ureg.meter)
    assert(has_uncertainty(x))

def read_csv(filename):
    with open(filename) as f:
        reader = csv.DictReader(f, delimiter=',')
        field_names = reader.fieldnames
        df = {}
        
        units = []
        short_field_names = []
        for field_name in field_names:
            unit = ''
            short_field_name = field_name
            if '(' in field_name:
                assert(')' in field_name)
                unit = field_name[field_name.index('(')+1:]
                unit = unit[0:unit.index(')')]
                short_field_name = field_name[0:field_name.index('(')-1].strip()
            
            units.append(unit)
            short_field_names.append(short_field_name)
        
        for row in reader:
            for field_name, short_field_name, unit in zip(field_names, short_field_names, units):
                if unit != 'str':
                    if row[field_name] == '':
                        value = np.nan
                    else:
                        value = float(row[field_name])
                    
                    is_float = True
                else:
                    value = row[field_name]
                    is_float = False
                
                if short_field_name in df:
                    if is_float:
                        df[short_field_name] = np.append(df[short_field_name], [value])
                    else:
                        df[short_field_name].append(value)
                else:
                    if is_float:
                        df[short_field_name] = np.array([value])
                    else:
                        df[short_field_name] = [value]
        
        for short_field_name, unit in zip(short_field_names, units):
            if unit != 'str':
                df[short_field_name] = ureg.Quantity(df[short_field_name], unit)
    
    # Make sure that all of the keys have the same length.
    for short_field_name in short_field_names:
        assert(len(df[short_field_name]) == len(df[short_field_names[0]]))
    
    return df

def test_read_csv():
    df = read_csv('data/test.csv')
    
    # Test that the correct number of rows is read.
    for key in df.keys():
        assert(len(df[key]) == 5)
    
    # Test that the correct number of cols is read.
    assert(len(df.keys()) == 4)
    
    # TODO: Test that the correct units are read (including str and bool).
    assert(df['L'].check('[length]'))
    assert(df['Re'].check(''))
    
    # Check that the numbers are as expected.
    assert(np.allclose(df['L'].magnitude, np.array([0.1, 0.2, 0.3, 0.4, 0.5])))
    assert(np.allclose(df['Re'].magnitude, np.array([100000, 100000, 100000, 100000, 100000])))
    assert(np.allclose(df['U'].magnitude, np.array([np.nan, 2, np.nan, 4, 7]), equal_nan=True))
    assert(df['classification'] == ['A', 'B', 'C', 'A', 'C'])
    
    # TODO: Check that str and filename columns are strings.
    
    # TODO: Check that filenames exist.
    
    # TODO: Test that data with uncertainties has uncertainties, and the uncertainties are correct.

def add_percent_uncertainty(arr, percent):
    assert(not(has_uncertainty(arr)))
    
    return_arr = np.array([])
    
    z = scipy.stats.norm.ppf(1 - (1 - interval_probability_level) / 2)
    multiplier = (percent / 100.) / z
    for value in arr:
        number = value.magnitude
        return_arr = np.append(return_arr, ufloat(number, number * multiplier))
    
    return_arr = ureg.Quantity(return_arr, arr.units)
    
    return return_arr

def add_absolute_uncertainty(arr, uncertainty):
    assert(not(has_uncertainty(arr)))
    
    return_arr = np.array([])
    
    z = scipy.stats.norm.ppf(1 - (1 - interval_probability_level) / 2)
    uncertainty_magnitude = uncertainty.magnitude / z
    for value in arr:
        number = value.magnitude
        return_arr = np.append(return_arr, ufloat(number, uncertainty_magnitude))
    
    return_arr = ureg.Quantity(return_arr, arr.units)
    
    return return_arr

# Configure Pint

ureg = pint.UnitRegistry(system='mks',  auto_reduce_dimensions=True)
ureg.setup_matplotlib()
with warnings.catch_warnings(): # Disable an annoying warning about deprecated behavior. Since I never did it the old way, I don't have to worry about this.
    warnings.simplefilter("ignore")
    ureg.Quantity([])

# <https://stackoverflow.com/a/51180482/1124489>
ureg.define('fraction = [] = frac')
ureg.define('percent = 1e-2 frac = pct')
ureg.define('ndm = []') # Non-DiMensional

# TODO: Add tests for all functions created so far.
# TODO: Add ability to handle an uncertainty column in the CSV file.
# TODO: Add the ability to handle a bool column in the CSV file. For example: screen: true/false
# TODO: Add the ability to handle a filename column in the CSV file. Check for the existence of the file.
# TODO: Add covariance data as (for example) Re_j0 will be correlated with d_0.
# TODO: Add docstrings and doctest.
