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
import math
import os.path

interval_probability_level = 0.95
z = scipy.stats.norm.ppf(1 - (1 - interval_probability_level) / 2)

# <https://stackoverflow.com/a/14981125/1124489>
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

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

def has_units(x):
    # This will work for floats, arrays, and ndarrays.
    
    try:
        # assume that this is an array or ndarray at first
        len(x) # This will fail if not an array or ndarray.
        
        for val in x:
            try:
                val.units
            except:
                return False
        
        return True
    except:
        try:
            x.units
            return True
        except:
            return False

def test_has_units():
    # float without units
    x = 0.20
    assert(not(has_units(x)))
    
    # float with units
    x = 0.20 * ureg.meter
    assert(has_units(x))
    
    # ndarray without units
    x = np.array([1., 2.])
    assert(not(has_units(x)))
    
    # ndarray with units
    x = ureg.Quantity(np.array([1., 2.]), ureg.meter)
    assert(has_units(x))

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
                if (unit == 'str') or (unit == 'filename'):
                    value = row[field_name]
                    is_float = False
                    
                    if (unit == 'filename') and (value != ''):
                        assert(os.path.isfile(value))
                elif unit == 'bool':
                    if row[field_name].lower() == 'true':
                        value = True
                    elif row[field_name].lower() == 'false':
                        value = False
                    else:
                        raise ValueError("Invalid boolean:", row[field_name])
                    
                    is_float = False
                else:
                    if row[field_name] == '':
                        value = np.nan
                    else:
                        value = float(row[field_name])
                    
                    is_float = True
                
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
            if (unit != 'str') and (unit != 'filename') and (unit != 'bool'):
                df[short_field_name] = ureg.Quantity(df[short_field_name], unit)
    
    # Make sure that all of the keys have the same length.
    for short_field_name in short_field_names:
        assert(len(df[short_field_name]) == len(df[short_field_names[0]]))
    
    # Add uncertainties if present.
    for short_field_name, unit in zip(short_field_names, units):
        if short_field_name.endswith(' uncertainty'):
            short_field_name_nominal = short_field_name.replace(' uncertainty', '').strip()
            assert(short_field_name_nominal in df.keys())
            arr = np.array([])
            for nominal, std_dev in zip(df[short_field_name_nominal], df[short_field_name]):
                nominal_magnitude = nominal.magnitude
                std_dev_magnitude = std_dev.magnitude
                arr = np.append(arr, ufloat(nominal_magnitude, std_dev_magnitude))
            
            df[short_field_name_nominal] = ureg.Quantity(arr, unit)
            del df[short_field_name]
        if short_field_name.endswith(' uncertainty %'):
            short_field_name_nominal = short_field_name.replace(' uncertainty %', '').strip()
            assert(short_field_name_nominal in df.keys())
            arr = np.array([])
            for nominal, percent_uncertainty in zip(df[short_field_name_nominal], df[short_field_name]):
                nominal_magnitude = nominal.magnitude
                std_dev_magnitude = nominal.magnitude * (percent_uncertainty.magnitude / 100.) / z
                arr = np.append(arr, ufloat(nominal_magnitude, std_dev_magnitude))
            
            df[short_field_name_nominal] = ureg.Quantity(arr, unit)
            del df[short_field_name]
    
    return df

def test_read_csv():
    df = read_csv('data/test.csv')
    
    # Test that the correct number of rows is read.
    for key in df.keys():
        assert(len(df[key]) == 5)
    
    # Test that the correct number of cols is read.
    assert(len(df.keys()) == 6)
    
    # Test that the correct units are read.
    assert(df['L'].check('[length]'))
    assert(df['Re'].check(''))
    
    # Test that data with uncertainties has uncertainties.
    assert(has_uncertainty(df['L']))
    assert(has_uncertainty(df['Re']))
    
    # Check that str and filename columns are strings, and that the bool columns are bools.
    for value in df['classification']:
        assert(isinstance(value, str))
    for value in df['photo']:
        assert(isinstance(value, str))
    for value in df['screen']:
        assert(isinstance(value, bool))
    
    # Check that the numbers are as expected, including both mean and uncertainties.
    assert(all_close_ud(df['L'], ureg.Quantity(unumpy.uarray([0.1, 0.2, 0.3, 0.4, 0.5], [0.05, 0.05, 0.05, 0.2, 0.2]), ureg.mm)))
    assert(all_close_ud(df['Re'], ureg.Quantity(unumpy.uarray([100000, 100000, 100000, 100000, 100000], (100000 / 1.959963984540054) * np.array([0.1, 0.1, 0.2, 0.2, 0.2])), ureg('ndm'))))
    assert(np.allclose(df['U'].magnitude, np.array([np.nan, 2, np.nan, 4, 7]), equal_nan=True))
    assert(df['screen'] == [False, False, True, True, False])
    assert(df['classification'] == ['A', 'B', 'C', 'A', 'C'])
    assert(df['photo'] == ['', '', '', 'data/asset_hydraulic_1951_fig_10.jpg', ''])
    
    # Check that filenames exist.
    for value in df['photo']:
        if value != '':
            assert(os.path.isfile(value))

def all_close_ud(arr1, arr2):
    assert(has_uncertainty(arr1))
    assert(has_units(arr1))
    assert(has_uncertainty(arr2))
    assert(has_units(arr2))
    
    arr1_magnitude_nominal = []
    arr1_magnitude_std_dev = []
    
    for value1 in arr1:
        number1 = value1.magnitude
        arr1_magnitude_nominal = np.append(arr1_magnitude_nominal, [number1.nominal_value])
        arr1_magnitude_std_dev = np.append(arr1_magnitude_std_dev, [number1.std_dev])
    
    arr2_magnitude_nominal = []
    arr2_magnitude_std_dev = []
    
    for value2 in arr2:
        number2 = value2.magnitude
        arr2_magnitude_nominal = np.append(arr2_magnitude_nominal, [number2.nominal_value])
        arr2_magnitude_std_dev = np.append(arr2_magnitude_std_dev, [number2.std_dev])
    
    return (np.allclose(arr1_magnitude_nominal, arr2_magnitude_nominal, equal_nan=True) and np.allclose(arr1_magnitude_std_dev, arr2_magnitude_std_dev, equal_nan=True))

def test_all_close_ud():
    assert(all_close_ud(ureg.Quantity(unumpy.uarray([0.1, 0.2, 0.3, 0.4, 0.5], [0.05, 0.05, 0.05, 0.2, 0.2]), ureg.mm), ureg.Quantity(unumpy.uarray([0.1, 0.2, 0.3, 0.4, 0.5], [0.05, 0.05, 0.05, 0.2, 0.2]), ureg.mm)))
    assert(not(all_close_ud(ureg.Quantity(unumpy.uarray([0.2, 0.2, 0.3, 0.4, 0.5], [0.05, 0.05, 0.05, 0.2, 0.2]), ureg.mm), ureg.Quantity(unumpy.uarray([0.1, 0.2, 0.3, 0.4, 0.5], [0.05, 0.05, 0.05, 0.2, 0.2]), ureg.mm))))
    assert(not(all_close_ud(ureg.Quantity(unumpy.uarray([0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.05, 0.05, 0.2, 0.2]), ureg.mm), ureg.Quantity(unumpy.uarray([0.1, 0.2, 0.3, 0.4, 0.5], [0.05, 0.05, 0.05, 0.2, 0.2]), ureg.mm))))

def add_percent_uncertainty(arr, percent):
    assert(not(has_uncertainty(arr)))
    assert(has_units(arr))
    
    return_arr = np.array([])
    
    multiplier = (percent / 100.) / z
    for value in arr:
        number = value.magnitude
        return_arr = np.append(return_arr, ufloat(number, number * multiplier))
    
    return_arr = ureg.Quantity(return_arr, arr.units)
    
    return return_arr

def test_add_percent_uncertainty():
    arr = ureg.Quantity(np.array([1., 2., 3.]), ureg.meter)
    assert(not(has_uncertainty(arr)))
    
    arr = add_percent_uncertainty(arr, 10.)
    
    # test that uncertainty is present
    assert(has_uncertainty(arr))
    
    # test that the amount of uncertainty is correct
    for value in arr:
        number = value.magnitude
        assert(math.isclose(number.std_dev, number.nominal_value * 0.1 / z))

def add_absolute_uncertainty(arr, uncertainty):
    assert(not(has_uncertainty(arr)))
    assert(has_units(arr))
    assert(has_units(uncertainty))
    
    return_arr = np.array([])
    
    uncertainty_magnitude = uncertainty.magnitude / z
    for value in arr:
        number = value.magnitude
        return_arr = np.append(return_arr, ufloat(number, uncertainty_magnitude))
    
    return_arr = ureg.Quantity(return_arr, arr.units)
    
    return return_arr

def test_add_absolute_uncertainty():
    arr = ureg.Quantity(np.array([1., 2., 3.]), ureg.meter)
    assert(not(has_uncertainty(arr)))
    
    arr = add_absolute_uncertainty(arr, 0.1 * ureg.meter)
    
    # test that uncertainty is present
    assert(has_uncertainty(arr))
    
    # test that the amount of uncertainty is correct
    for value in arr:
        number = value.magnitude
        assert(math.isclose(number.std_dev, 0.1 / z))

def create_table_query_start(table_name):
    return 'CREATE TABLE '+table_name+' ('

def create_table_query_column(column_name, datatype, primary_key=False, not_null=False, lower_bound=None, upper_bound=None, end=False):
    assert(datatype.lower() in {'integer', 'real', 'text'})
    
    return_string = '\n'+column_name+' '+datatype
    
    if primary_key:
        return_string += ' PRIMARY KEY'
    
    if not_null:
        return_string += ' NOT NULL'
    
    if not(lower_bound is None) or not(upper_bound is None):
        return_string += ' CHECK '
        if not(lower_bound is None) and not(upper_bound is None):
            # TODO: Assert that lower_bound and upper_bound are floats.
            return_string += '(('+column_name+' > '+str(lower_bound)+') and ('+column_name+' < '+str(upper_bound)+'))'
        elif not(lower_bound is None):
            return_string += '('+column_name+' > '+str(lower_bound)+')'
        elif not(upper_bound is None):
            return_string += '('+column_name+' < '+str(upper_bound)+')'
        else:
            raise ValueError("Invalid bounds?")
    
    return_string += ','
    
    if end:
        return_string += '\n) STRICT;'
    
    return return_string

def test_create_table_query():
    create_query = create_table_query_start('jetbreakup')
    create_query += create_table_query_column('id', 'integer', primary_key=True)
    create_query += create_table_query_column('We_j0', 'real', not_null=True, lower_bound=0)
    create_query += create_table_query_column('Re_j0', 'real', not_null=True, lower_bound=0, upper_bound=1e8)
    create_query += create_table_query_column('Tubar_0', 'real', not_null=True, upper_bound=1, end=True)
    
    assert(create_query.startswith('CREATE TABLE '))
    assert(create_query.endswith(') STRICT;'))
    assert('PRIMARY KEY' in create_query)
    assert(create_query.count('(') == create_query.count(')'))

# def create_db(filename):
    # try:
        # con = sqlite3.connect(db_file)
    # except Error as e:
        # eprint(e)
        # exit(-1)
    
    # c = con.cursor()

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

# TODO: Write wrapper functions so that you can switch out Pint and uncertainties later if you want to.
# TODO: Add docstrings.
# TODO: Add uncertainty columns. uncertainty >= 0
# TODO: Add dimensions and LaTeX string to an info table.
# TODO: Add covariance column. Constraint: <https://math.stackexchange.com/a/3830254>
