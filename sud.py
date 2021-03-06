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
import json
import sqlite3

interval_probability_level = 0.95
z = scipy.stats.norm.ppf(1 - (1 - interval_probability_level) / 2)

# <https://stackoverflow.com/a/14981125/1124489>
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def is_float(x):
    if isinstance(x, str):
        return False
    
    try:
        float(x)
        return True
    except:
        return False

def test_is_float():
    assert is_float(1.1), "check that the function works right on floats"
    assert not(is_float('a')), "check that the function works right on strings"

def str_is_float(x):
    assert isinstance(x, str), "function input must be a string"
    try:
        float(x)
        return True
    except:
        return False

def test_str_is_float():
    assert str_is_float('1.1'), "check that the function works right on floats"
    assert not(str_is_float('a')), "check that the function works right on strings"

def has_uncertainty(x):
    # This will work for floats, arrays, ndarrays, strs, and bools.
    
    if isinstance(x, str) or isinstance(x, bool):
        return False
    
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
    assert not(has_uncertainty(x))
    
    # float with uncertainty
    x = ufloat(0.20, 0.01)
    assert has_uncertainty(x)
    
    # ndarray without uncertainty
    x = np.array([1., 2.])
    assert not(has_uncertainty(x))
    
    # ndarray with uncertainty
    x = unumpy.uarray([1, 2], [0.01, 0.002])
    assert has_uncertainty(x)
    
    # float with units without uncertainty
    x = 0.20 * ureg.meter
    assert not(has_uncertainty(x))
    
    # float with units with uncertainty
    x = ufloat(0.20, 0.01) * ureg.meter
    assert has_uncertainty(x)
    
    # ndarray with units without uncertainty
    x = ureg.Quantity(np.array([1., 2.]), ureg.meter)
    assert not(has_uncertainty(x))
    
    # ndarray with units with uncertainty
    x = ureg.Quantity(unumpy.uarray([1, 2], [0.01, 0.002]), ureg.meter)
    assert has_uncertainty(x)
    
    # string
    assert not(has_uncertainty('test'))
    
    # bool
    assert not(has_uncertainty(True))

def has_units(x):
    # This will work for floats, arrays, ndarrays, strs, and bools.
    
    if isinstance(x, str) or isinstance(x, bool):
        return False
    
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
    assert not(has_units(x))
    
    # float with units
    x = 0.20 * ureg.meter
    assert has_units(x)
    
    # ndarray without units
    x = np.array([1., 2.])
    assert not(has_units(x))
    
    # ndarray with units
    x = ureg.Quantity(np.array([1., 2.]), ureg.meter)
    assert has_units(x)

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
                assert ')' in field_name
                unit = field_name[field_name.index('(')+1:]
                unit = unit[0:unit.index(')')]
                short_field_name = field_name[0:field_name.index('(')-1].strip()
            
            units.append(unit)
            short_field_names.append(short_field_name)
        
        for row in reader:
            for field_name, short_field_name, unit in zip(field_names, short_field_names, units):
                if (unit == 'str') or (unit == 'filename'):
                    value = row[field_name]
                    var_is_float = False
                    
                    if (unit == 'filename') and (value != ''):
                        assert os.path.isfile(value)
                elif unit == 'bool':
                    if row[field_name].lower() == 'true':
                        value = True
                    elif row[field_name].lower() == 'false':
                        value = False
                    else:
                        raise ValueError("Invalid boolean:", row[field_name])
                    
                    var_is_float = False
                else:
                    if row[field_name] == '':
                        value = np.nan
                    else:
                        value = float(row[field_name])
                    
                    var_is_float = True
                
                if short_field_name in df:
                    if var_is_float:
                        df[short_field_name] = np.append(df[short_field_name], [value])
                    else:
                        df[short_field_name].append(value)
                else:
                    if var_is_float:
                        df[short_field_name] = np.array([value])
                    else:
                        df[short_field_name] = [value]
        
        for short_field_name, unit in zip(short_field_names, units):
            if (unit != 'str') and (unit != 'filename') and (unit != 'bool'):
                df[short_field_name] = ureg.Quantity(df[short_field_name], unit)
    
    # Make sure that all of the keys have the same length.
    for short_field_name in short_field_names:
        assert len(df[short_field_name]) == len(df[short_field_names[0]])
    
    # Add uncertainties if present.
    for short_field_name, unit in zip(short_field_names, units):
        if short_field_name.endswith(' uncertainty'):
            short_field_name_nominal = short_field_name.replace(' uncertainty', '').strip()
            assert short_field_name_nominal in df.keys()
            arr = np.array([])
            for nominal, std_dev in zip(df[short_field_name_nominal], df[short_field_name]):
                nominal_magnitude = nominal.magnitude
                std_dev_magnitude = std_dev.magnitude
                arr = np.append(arr, ufloat(nominal_magnitude, std_dev_magnitude))
            
            df[short_field_name_nominal] = ureg.Quantity(arr, unit)
            del df[short_field_name]
        if short_field_name.endswith(' uncertainty %'):
            short_field_name_nominal = short_field_name.replace(' uncertainty %', '').strip()
            assert short_field_name_nominal in df.keys()
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
        assert len(df[key]) == 5
    
    # Test that the correct number of cols is read.
    assert len(df.keys()) == 6
    
    # Test that the correct units are read.
    assert df['L'].check('[length]')
    assert df['Re'].check('')
    
    # Test that data with uncertainties has uncertainties.
    assert has_uncertainty(df['L'])
    assert has_uncertainty(df['Re'])
    
    # Check that str and filename columns are strings, and that the bool columns are bools.
    for value in df['classification']:
        assert isinstance(value, str)
    for value in df['photo']:
        assert isinstance(value, str)
    for value in df['screen']:
        assert isinstance(value, bool)
    
    # Check that the numbers are as expected, including both mean and uncertainties.
    assert all_close_ud(df['L'], ureg.Quantity(unumpy.uarray([0.1, 0.2, 0.3, 0.4, 0.5], [0.05, 0.05, 0.05, 0.2, 0.2]), ureg.mm))
    assert all_close_ud(df['Re'], ureg.Quantity(unumpy.uarray([100000, 100000, 100000, 100000, 100000], (100000 / 1.959963984540054) * np.array([0.1, 0.1, 0.2, 0.2, 0.2])), ureg('ndm')))
    assert np.allclose(df['U'].magnitude, np.array([np.nan, 2, np.nan, 4, 7]), equal_nan=True)
    assert df['screen'] == [False, False, True, True, False]
    assert df['classification'] == ['A', 'B', 'C', 'A', 'C']
    assert df['photo'] == ['', '', '', 'data/asset_hydraulic_1951_fig_10.jpg', '']
    
    # Check that filenames exist.
    for value in df['photo']:
        if value != '':
            assert os.path.isfile(value)

def all_close_ud(arr1, arr2):
    assert has_uncertainty(arr1)
    assert has_units(arr1)
    assert has_uncertainty(arr2)
    assert has_units(arr2)
    
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
    assert all_close_ud(ureg.Quantity(unumpy.uarray([0.1, 0.2, 0.3, 0.4, 0.5], [0.05, 0.05, 0.05, 0.2, 0.2]), ureg.mm), ureg.Quantity(unumpy.uarray([0.1, 0.2, 0.3, 0.4, 0.5], [0.05, 0.05, 0.05, 0.2, 0.2]), ureg.mm))
    assert not(all_close_ud(ureg.Quantity(unumpy.uarray([0.2, 0.2, 0.3, 0.4, 0.5], [0.05, 0.05, 0.05, 0.2, 0.2]), ureg.mm), ureg.Quantity(unumpy.uarray([0.1, 0.2, 0.3, 0.4, 0.5], [0.05, 0.05, 0.05, 0.2, 0.2]), ureg.mm)))
    assert not(all_close_ud(ureg.Quantity(unumpy.uarray([0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.05, 0.05, 0.2, 0.2]), ureg.mm), ureg.Quantity(unumpy.uarray([0.1, 0.2, 0.3, 0.4, 0.5], [0.05, 0.05, 0.05, 0.2, 0.2]), ureg.mm)))

def add_percent_uncertainty(arr, percent):
    assert not(has_uncertainty(arr))
    assert has_units(arr)
    
    return_arr = np.array([])
    
    multiplier = (percent / 100.) / z
    for value in arr:
        number = value.magnitude
        if np.isnan(number):
            return_arr = np.append(return_arr, ufloat(number, np.nan))
        else:
            return_arr = np.append(return_arr, ufloat(number, number * multiplier))
    
    return_arr = ureg.Quantity(return_arr, arr.units)
    
    return return_arr

def test_add_percent_uncertainty():
    arr = ureg.Quantity(np.array([1., 2., 3.]), ureg.meter)
    assert not(has_uncertainty(arr))
    
    arr = add_percent_uncertainty(arr, 10.)
    
    # test that uncertainty is present
    assert has_uncertainty(arr)
    
    # test that the amount of uncertainty is correct
    for value in arr:
        number = value.magnitude
        assert math.isclose(number.std_dev, number.nominal_value * 0.1 / z)

def add_absolute_uncertainty(arr, uncertainty):
    assert not(has_uncertainty(arr))
    assert has_units(arr)
    assert has_units(uncertainty)
    
    return_arr = np.array([])
    
    uncertainty_magnitude = uncertainty.magnitude / z
    for value in arr:
        number = value.magnitude
        if np.isnan(number):
            return_arr = np.append(return_arr, ufloat(number, np.nan))
        else:
            return_arr = np.append(return_arr, ufloat(number, uncertainty_magnitude))
    
    return_arr = ureg.Quantity(return_arr, arr.units)
    
    return return_arr

def test_add_absolute_uncertainty():
    arr = ureg.Quantity(np.array([1., 2., 3.]), ureg.meter)
    assert not(has_uncertainty(arr))
    
    arr = add_absolute_uncertainty(arr, 0.1 * ureg.meter)
    
    # test that uncertainty is present
    assert has_uncertainty(arr)
    
    # test that the amount of uncertainty is correct
    for value in arr:
        number = value.magnitude
        assert math.isclose(number.std_dev, 0.1 / z)

def create_db(table_name):
    # TODO: After updating to SQlite 3.37.0 or later, add STRICT keyword. <https://www.sqlite.org/stricttables.html>
    
    assert len(table_name) < 24
    
    data_create_query = f"CREATE TABLE {table_name} (\n{table_name}_id integer PRIMARY KEY,"
    
    info_create_query = f"CREATE TABLE {table_name}_info (\n{table_name}_info_id integer PRIMARY KEY,\nvariable text NOT NULL,\nunits text NOT NULL,\nlatex text,\ndescription text NOT NULL);"
    
    info_insert_queries    = []
    info_insert_params_arr = []
    
    with open('data/'+table_name+'.json') as f:
        data = json.load(f)
        
        for variable in data['variables']:
            assert 'variable' in variable.keys(), "all variables must have a variable name"
            print("Reading variable:", variable['variable'])
            assert not(' ' in variable['variable']), "no spaces in variable names"
            
            # Data validation
            
            assert 'datatype' in variable.keys()
            assert 'description' in variable.keys()
            
            assert variable['datatype'] in {'int', 'float', 'str', 'bool'}
            
            if variable['datatype'] == 'float':
                variable['datatype'] = 'real'
                assert 'units' in variable.keys()
                assert len(variable['units']) > 0
                assert 'latex' in variable.keys()
                assert len(variable['latex']) > 0
            elif variable['datatype'] == 'int':
                assert 'units' in variable.keys()
                assert len(variable['units']) > 0
                assert 'latex' in variable.keys()
                assert len(variable['latex']) > 0
            
            if (variable['datatype'] == 'str') or (variable['datatype'] == 'bool'):
                assert not('units' in variable.keys())
            
            if not('units' in variable.keys()):
                if variable['datatype'] == 'str':
                    variable['units'] = 'str'
                elif variable['datatype'] == 'bool':
                    variable['units'] = 'bool'
            
            assert 'units' in variable.keys()
            
            if variable['datatype'] == 'str':
                variable['datatype'] = 'text'
            elif variable['datatype'] == 'bool':
                variable['datatype'] = 'int'
            
            assert len(variable['description']) > 0
            
            if 'lower_bound' in variable.keys():
                # If lower_bound defined, then datatype must be real.
                assert variable['datatype'] == 'real'
                
                # If lower_bound defined, then the number must be a float.
                assert is_float(variable['lower_bound'])
            
            if 'upper_bound' in variable.keys():
                # If upper_bound defined, then datatype must be real.
                assert variable['datatype'] == 'real'
                
                # If upper_bound defined, then the number must be a float.
                assert is_float(variable['upper_bound'])
                
                # If both upper and lower bounds are present, check that the lower is lower than the upper.
                if 'lower_bound' in variable.keys():
                    assert variable['lower_bound'] < variable['upper_bound']
            
            if 'not_null' in variable.keys():
                assert isinstance(variable['not_null'], bool)
            else:
                variable['not_null'] = False
            
            if 'set' in variable.keys():
                assert len(variable['set']) > 1, "the number of set members must be greater than 1"
            
            # construct the query
            
            data_create_query += f"\n{variable['variable']} {variable['datatype']}"
            
            if variable['not_null']:
                data_create_query += " NOT NULL"
            
            if ('lower_bound' in variable.keys()) or ('upper_bound' in variable.keys()):
                data_create_query += ' CHECK '
                if ('lower_bound' in variable.keys()) and ('upper_bound' in variable.keys()):
                    data_create_query += f"(({variable['variable']} > {variable['lower_bound']}) and ({variable['variable']} < {variable['upper_bound']}))"
                elif ('lower_bound' in variable.keys()):
                    data_create_query += f"({variable['variable']} > {variable['lower_bound']})"
                elif ('upper_bound' in variable.keys()):
                    data_create_query += f"({variable['variable']} < {variable['upper_bound']})"
                else:
                    raise ValueError("Invalid bounds? This should be impossible.")
            
            data_create_query += ","
            
            # Only add uncertainties if variable is real.
            if variable['datatype'] == 'real':
                data_create_query += f"\n{variable['variable']}_std_dev real"
                
                if variable['not_null']:
                    data_create_query += " NOT NULL"
                
                data_create_query += f" CHECK ({variable['variable']}_std_dev > 0)"
                
                data_create_query += ','
            
            if 'latex' in variable.keys():
                info_insert_queries.append(f"INSERT INTO {table_name}_info (variable, units, latex, description) VALUES(?, ?, ?, ?);")
                info_insert_params_arr.append((variable['variable'], variable['units'], variable['latex'], variable['description']))
            else:
                info_insert_queries.append(f"INSERT INTO {table_name}_info (variable, units, description) VALUES(?, ?, ?);")
                info_insert_params_arr.append((variable['variable'], variable['units'], variable['description']))
    
    data_create_query = data_create_query[:-1]
    
    data_create_query += "\n);"
    #data_create_query += "\n) STRICT;"
    
    assert not('variable[' in data_create_query)
    assert not('variable[' in info_create_query)
    
    assert data_create_query.startswith('CREATE TABLE ')
    assert data_create_query.endswith(');')
    #assert data_create_query.endswith(') STRICT;')
    assert 'PRIMARY KEY' in data_create_query
    assert data_create_query.count('(') == data_create_query.count(')')
    
    assert info_create_query.startswith('CREATE TABLE ')
    assert info_create_query.endswith(');')
    #assert info_create_query.endswith(') STRICT;')
    assert 'PRIMARY KEY' in info_create_query
    assert info_create_query.count('(') == info_create_query.count(')')
    
    # Assert that number of question marks equals length of params.
    for info_insert_query, info_insert_params in zip(info_insert_queries, info_insert_params_arr):
        assert info_insert_query.count('?') == len(info_insert_params)
    
    db_file = f"data/{table_name}.sqlite"
    
    try:
        if os.path.exists(db_file):
           os.remove(db_file)
        con = sqlite3.connect(db_file)
    except Error as e:
        eprint(e)
        exit(-1)
    
    c = con.cursor()
    
    print(data_create_query)
    c.execute(data_create_query)
    
    print(info_create_query)
    c.execute(info_create_query)
    
    print(info_insert_queries)
    print(info_insert_params_arr)
    for info_insert_query, info_insert_params in zip(info_insert_queries, info_insert_params_arr):
        c.execute(info_insert_query, info_insert_params)
    
    con.commit()
    #con.close()
    
    return con

def insert_data(con, df):
    c = con.cursor()
    
    # Get a list of all variables available.
    
    variables = []
    c.execute("select * from sqlite_master;")
    query_result = c.fetchall()
    assert len(query_result) == 2
    table_name = query_result[0][1]
    c.execute(f"pragma table_info({table_name});")
    query_result = c.fetchall()
    for info in query_result:
        if not(info[1].endswith('_std_dev')) and not(info[1].endswith(f"{table_name}_id")):
            variables.append(info[1])
    
    # Check that all variables in the database are present in the dataframe.
    for variable in df.keys():
        assert variable in variables
    
    # Add data to database.
    rows = len(df[list(df.keys())[0]])
    for i in range(rows):
        insert_query = f"INSERT INTO {table_name} ("
        insert_params = ()
        
        for variable in df.keys():
            insert_query += f"{variable}, "
            #print(variable, df[variable][i])
            
            if has_units(df[variable][i]):
                variable_magnitude = df[variable][i].magnitude
                
                # TODO: Check that units match.
                # TODO: For variables which can be null, if there is a mean, require a std_dev too. Sqlite by itself will not be able to enforce that.
                
                if has_uncertainty(df[variable][i]):
                    # For variables with uncertainty, split into mean and std_dev and insert both.
                    variable_nominal = variable_magnitude.nominal_value
                    variable_std_dev = variable_magnitude.std_dev
                    
                    if np.isnan(variable_nominal):
                        variable_nominal = None
                    
                    if np.isnan(variable_std_dev):
                        variable_std_dev = None
                    
                    insert_query += f"{variable}_std_dev, "
                    insert_params += (variable_nominal, variable_std_dev)
                else:
                    insert_params += (variable_magnitude,)
            else:
                if isinstance(df[variable][i], str):
                    variable = df[variable][i]
                    
                    if variable == '':
                        variable = None
                    
                    # TODO: For string data, check against set from JSON file when adding to database.
                else:
                    variable = df[variable][i]
                
                insert_params += (variable,)
        
        insert_query = insert_query[:-2] + ") VALUES("
        
        # Add the appropriate number of question marks to parameterize the query.
        for param in insert_params:
            insert_query += '?, '
        
        insert_query = insert_query[:-2] + ");"
        
        assert insert_query.count('?') == len(insert_params)
        
        print(insert_query)
        print(insert_params)
        c.execute(insert_query, insert_params)

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

con = create_db('test')

df = read_csv('data/test.csv')

df['U'] = add_absolute_uncertainty(df['U'], 0.1 * ureg.meter / ureg.second)

insert_data(con, df)

con.close()
