import numpy as np

__version__ = '0.0.1'


class DataFrame:

    def __init__(self, data):
        """
        A DataFrame holds two dimensional heterogeneous data. Create it by
        passing a dictionary of NumPy arrays to the values parameter

        Parameters
        ----------
        data: dict
            A dictionary of strings mapped to NumPy arrays. The key will
            become the column name.
        """
        # check for correct input types
        self._check_input_types(data)

        # check for equal array lengths
        self._check_array_lengths(data)

        # convert unicode arrays to object
        # _data will be used from now on instead of data defined in the init method
        self._data = self._convert_unicode_to_object(data)

        # Allow for special methods for strings
        # we are assigning the attribute of self a new instance of stringmethods class
        # self argument here will be passed to df parameter of the __init__ in the string methods    
        self.str = StringMethods(self)
        self._add_docs()

        print('init method')

#isinstance is method that checks whether an object of a particular type
    def _check_input_types(self, data):
        if not isinstance(data, dict):
            raise TypeError(f'$data must be dictionary')

        for key,values in data.items():
            if not isinstance(key, str):
                raise TypeError('keys must be string')
            if not isinstance(values, np.ndarray):
                raise TypeError('values must be 1 D arrays')
            else:
                if values.ndim != 1:
                    raise ValueError('Each value must be a 1-D NumPy array')
                


    def _check_array_lengths(self, data):
        for i, values in enumerate(data.items()):
            if i == 0:
                length = len(value)
            elif length != len(value):
                raise ValueError("all the arrays must be of the samw length")    

    def _convert_unicode_to_object(self, data):
        new_data = {}
        for key, values in data.items():
            if values.dtype.kind == 'U':
                new_data(key) = values.astype('O')
            else:
                new_data[key] = values
        return new_data

    def __len__(self):
        """
        Make the builtin len function work with our dataframe

        Returns
        -------
        int: the number of rows in the dataframe
        """
        return len(next(iter((self._data))))

    #getter method
    @property
    def columns(self):
        """
        _data holds column names mapped to arrays
        take advantage of internal ordering of dictionaries to
        put columns in correct order in list. Only works in 3.6+

        Returns
        -------
        list of column names
        """
         # returning the data will return the keys, we need to explicitly give values and items to retrieve them
        return list(self._data)

    #setter method
    @columns.setter
    def columns(self, columns):
        """
        Must supply a list of columns as strings the same length
        as the current DataFrame

        Parameters
        ----------
        columns: list of strings

        Returns
        -------
        None
        """
        new_data = {}
        if not isinstance(columns, list):
            raise TypeError("not a list")
        if len(columns) != len(self._data):
            raise ValueError("the number of columns does not match")
        else:
            for cols in columns:
                if not isinstance(cols, str):
                    raise ValueError(" columns are not strings")        
        if len(self._data) != len(set(columns)):
            raise ValueError("there are duplicated string in the columns")
        # zip will turn it into a tuple which is then converted into a list    
        new_data = dict(zip(columns, self._data.values()))    
        self._data = new_data
        

    @property
    def shape(self):
        """
        Returns
        -------
        two-item tuple of number of rows and columns
        """
        return len(self), len(self._data)

    def _repr_html_(self):
        """
        Used to create a string of HTML to nicely display the DataFrame
        in a Jupyter Notebook. Different string formatting is used for
        different data types.

        The structure of the HTML is as follows:
        <table>
            <thead>
                <tr>
                    <th>data</th>
                    ...
                    <th>data</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>{i}</strong></td>
                    <td>data</td>
                    ...
                    <td>data</td>
                </tr>
                ...
                <tr>
                    <td><strong>{i}</strong></td>
                    <td>data</td>
                    ...
                    <td>data</td>
                </tr>
            </tbody>
        </table>
        """
        pass

    @property
    def values(self):
        """
        Returns
        -------
        A single 2D NumPy array of the underlying data
        """
        return np.column_stack(list(self._data.values()))
    

    #Difficult level of programming 
    @property
    def dtypes(self):
        """
        Returns
        -------
        A two-column DataFrame of column names in one column and
        their data type in the other
        """
        DTYPE_NAME = {'O': 'string', 'i': 'int', 'f': 'float', 'b': 'bool'}
        col_names = np.array(list(self._data.keys()))
        # DTYPE_NAME[values.dtype.kind] will actually give values, following is a list comprehension 
        # see the final code which is much simpler
        dtypes = [DTYPE_NAME[values.dtype.kind] for values in self._data.values()]
        dtypes = np.array(dtypes)  
        new_data = {'Column Names':col_names,'Data Types': dtypes}

        return DataFrame(new_data)
        

    def __getitem__(self, item):
        """
        Use the brackets operator to simultaneously select rows and columns
        A single string selects one column -> df['colname']
        A list of strings selects multiple columns -> df[['colname1', 'colname2']]
        A one column DataFrame of booleans that filters rows -> df[df_bool]
        Row and column selection simultaneously -> df[rs, cs]
            where cs and rs can be integers, slices, or a list of integers
            rs can also be a one-column boolean DataFrame

        Returns
        -------
        A subset of the original DataFrame
        """
        # if isinstance(item, str):
        #     is_present = False
        #     for item in self._data.keys():
        #         is_present = True
        #         if is_present:
        #             return DataFrame({item: self._data[item]})
        #         else:
        #         ValueError("Column not present in the Dataframe")    
        # else:
        #     raise TypeError("Column entered is not string")   

        if isinstance(item, str):
            return DataFrame({item: self._data[item]})

        # Difficult programming
        # all values filtering of specific 
        if isinstance(item, list):
            return DataFrame({col: self._data[col] for col in item})

        if isinstance(item, DataFrame):
            #s shape[1] means the rows of dataframes shape[2] means columns of dataframe
            if item.shape[1] != 1:
                raise ValueError("Item must be a one column Dataframe")  
            arr = next(iter(item._data.values()))
            if arr.dtype.kind != 'b':
                raise ValueError("this is wrong")
            # values filtering according to the condition, self._data.items() means we are iterating through our current data
            #numpy is doing value[arr], which boolean selection/filtering of values, values is 1D array and arr is boolean array
            new_data = {}
            for col, values in self._data.items():
                 new_data[col] = values[arr]
                 return DataFrame(new_data)
           

            if isinstance(item, tuple):
                return self._getitem_tuple(item)

            raise TypeError("the item should be strinng, list, dataframe or tuple")    
              

    def _getitem_tuple(self, item):
        # simultaneous selection of rows and cols -> df[rs, cs]
        if len(item) != 2:
            raise ValueError("the length of the tuple entered must be 2")

        row_selection, col_selection = item

        if isinstance(row_selection, int):
            row_selection = [row_selection]
        #row_selection must be a one column dataframe with boolean values 
        elif isinstance(row_selection, DataFrame):
            if row_selection.shape[1] =! 1:
                raise TypeError("DataFrame must be one column") 
            #row_selection is now a numpy array
            row_selection = next(iter(row_selection._data.values()))   
            if row_selection.dtype.kind == 'b':
                raise TypeError("Row Selection Dataframe must be a boolean")
        #list and slice will be auto dealt with the underlying numpy arrays as it is the functions of numpy arrays    
        elif not isinstance(row_selection, (list,slice)): 
            raise TypeError("Row selection must be int, list,slice and Dataframe")
        
        if isinstance(col_selection, int):
            col_selection = [self.columns[col_selection]] # self.columns is giving out a list of string columns and col_selection which is integer is selecting a particular string of that list 
        elif isinstance(col_selection, str):
            col_selection = [col_selection] 
        elif isinstance(col_selection, list):
            new_col_selection = []
            for col in col_selection:
                if isinstance(col, int): #checking if there is any integer values in the list
                    new_col_selection = append(self.columns[col])
                else:
                    new_col_selection = append(col) #Columns entered is a string
            col_selection = new_col_selection                 
        elif isinstance(col_selection, slice):
            start = col_selection.start
            stop = col_selection.stop
            step = col_selection.step   

            #if start and stop are strings we are changing them to integers
            if isinstance(start, str):
                start = self.columns.index(start)

            if isinstance(stop, str):
                stop = self.columns.index(stop) + 1

            col_selection = self.columns[start:stop:step] # that is why we made the col_selection as a list   
                   

        else:
            raise TypeError("column selection must be str,list,int or slice")    

        new_data = {}

        for col in col_selection:
            new_data[col] = self._data[col][row_selection] #"[]"([row_selection]) can be used with underlying numpy array(self._data[col]) of the dictionary          

        return DataFrame(new_data)


    def _ipython_key_completions_(self):
        # allows for tab completion when doing df['c
        return self.columns

    def __setitem__(self, key, value):
        # adds a new column or a overwrites an old column
        if not isinstance(key, str):
            raise NotImplementedError("Columns should be string")
        #value should be 1 D numpy array with same dataframe length
        if not isinstance(value, ndarray):
            if value.ndim != 1:
                raise ValueError("The columns values are not 1 D")
            if len(value) =! len(self):
                raise("Columns length not matching the dataframe")

        elif isinstance(value, Dataframe):
            if value.shape[1] != 1:
                raise ValueError("Columns values cannot be dataframe")
            if len(value) =! len(self):
                raise("Columns length not matching the dataframe")
            value = next(iter(value._data.values()))    
        elif isinstance(value, (int,str,bool,float)):
            value = np.repeat(value, len(self))
        else:
            raise TypeError("Setting dataframe must be python string dictionary with numpy arrays")
        

        if value.dtype.kind == 'U':
            value = value.astype('O')
          #appending the new key or exsiting, values in the dictionary 
        self._data[key] = value    
            


    

    def head(self, n=5):
        """
        Return the first n rows

        Parameters
        ----------
        n: int

        Returns
        -------
        DataFrame
        """
        #take advantage of the get item we implemented
        return self[:n, :]


    def tail(self, n=5):
        """
        Return the last n rows

        Parameters
        ----------
        n: int
        
        Returns
        -------
        DataFrame
        """
     return self[-n:, :]

    #### Aggregation Methods ####

    def min(self):
        return self._agg(np.min)

    def max(self):
        return self._agg(np.max)

    def mean(self):
        return self._agg(np.mean)

    def median(self):
        return self._agg(np.median)

    def sum(self):
        return self._agg(np.sum)

    def var(self):
        return self._agg(np.var)

    def std(self):
        return self._agg(np.std)

    def all(self):
        return self._agg(np.all)

    def any(self):
        return self._agg(np.any)

    def argmax(self):
        return self._agg(np.argmax)

    def argmin(self):
        return self._agg(np.argmin)

    def _agg(self, aggfunc):
        """
        Generic aggregation function that applies the
        aggregation to each column

        Parameters
        ----------
        aggfunc: str of the aggregation function name in NumPy
        
        Returns
        -------
        A DataFrame
        """
        new_data = {}
        for col, values in self._data.items():
            try:
                new_data[col] = np.ndarray([aggfunc(values)])
            except:
                TypeError("The strings cannot be aggregated")
                continue

        return DataFrame(new_data)        
            

    def isna(self):
        """
        Determines whether each value in the DataFrame is missing or not, if missing then true if not missing then false

        Returns
        -------
        A DataFrame of booleans the same size as the calling DataFrame
        """
        new_data = {}
        for col, values in self._data.items():
            if values.dtype.kind == 'O':
                new_data[col] = values == None #in case of string values == None will output true if there is None values in string column
            else:
                #isnan will output true if there is a Nan value in the integer column
                new_data[col] = np.isnan(values)
        return DataFrame(new_data)            


    def count(self):
        """
        Counts the number of non-missing values per column

        Returns
        -------
        A DataFrame
        """
        # returns the dataframe containing true(1) values if the values are missing otherwise false(0)
        df = self.isna() 
        new_data = {}
        length = len(self) # returns the length of the self's dictionary property 
        for col, values in df._data.items():
#sum is actually counting the true(1) values which means it is counting the number of NaN or None items
#subracting the sum from total length means getting the count of non-Missing values
            new_data[col] = np.array([length - values.sum()]) 
        return DataFrame(new_data)   


    def unique(self):
        """
        Finds the unique values of each column

        Returns
        -------
        A list of one-column DataFrames
        """
        dfs = []
        for col, values in self._data.items():
            new_data = {col:np.unique(values)}
            dfs.append(DataFrame(new_data))
        # if there is single column , just return the dataframe otherwise return the entire list 
        # numpy unique function does not work if we have none values in the columns   
        if len(dfs) == 1:
            return dfs[0]

        return dfs   


    def nunique(self):
        """
        Find the number of unique values in each column

        Returns
        -------
        A DataFrame
        """
        new_data = {}
        for col, values in self._data.items():
            new_data[col] = np.array([len(np.unique(values))])
        return DataFrame(new_data)



    def value_counts(self, normalize=False):
        """
        Returns the frequency of each unique value for each column

        Parameters
        ----------
        normalize: bool
            If True, returns the relative frequencies (percent)

        Returns
        -------
        A list of DataFrames or a single DataFrame if one column
        """
        #Negative sign means greatest to least values as the values will become negative
        #then sorting positions will be assgined by argsort
        #Count column will have the frequency of each of those columns
        dfs = []
        for col, values in self._data.items():
            unique , counts = np.unique(value, return_counts = True)
            order = np.argsort(-counts)
            unique = unique[order]
            counts = counts[order]
            new_data = {col: unique, 'count': counts}
            dfs.append(DataFrame(new_data))
            return dfs
        if len(dfs) == 1:
            return dfs[0]

         return dfs      

    def rename(self, columns):
        """
        Renames columns in the DataFrame

        Parameters
        ----------
        columns: dict
            A dictionary mapping the old column name to the new column name
        
        Returns
        -------
        A DataFrame
        """
        if not isinstance(columns, dict):
            raise TypeError("columns must be a string")

        new_data = {}
        for col, values in self._data.items():
    #the get function returns the value for given key otherwisre it returns None
    # here the get function will return the value(new column) of the key(old column) -- {'old column name': 'new column name'}
    # if there is no value then the 2nd parameter will return  the old column- refer geeks for geeks        
            new_col = columns.get(col,col)
            new_data[new_col] = values

        return DataFrame(new_data)    

    def drop(self, columns):
        """
        Drops one or more columns from a DataFrame

        Parameters
        ----------
        columns: str or list of strings

        Returns
        -------
        A DataFrame
        """
        if is isinstance(columns, str):
            columns = [columns]
        elif not isinstance(columns, list):
            raise TypeError("Columns must be either string or list")    


        new_data = {}
        for col, values in self._data.items():
            #as we are trying to remove the columns passed to the drop function
            if not col in columns:
                new_data[col] = values

            return DataFrame(new_data)    



        return DataFrame(new_data)    
         

    #### Non-Aggregation Methods ####

    def abs(self):
        """
        Takes the absolute value of each value in the DataFrame

        Returns
        -------
        A DataFrame
        """
        return self._non_agg(np.abs)

    def cummin(self):
        """
        Finds cumulative minimum by column

        Returns
        -------
        A DataFrame
        """
        return self._non_agg(np.minimum.accumulate)

    def cummax(self):
        """
        Finds cumulative maximum by column

        Returns
        -------
        A DataFrame
        """
        return self._non_agg(np.maximum.accumulate)

    def cumsum(self):
        """
        Finds cumulative sum by column

        Returns
        -------
        A DataFrame
        """
        return self._non_agg(np.cumsum)

    def clip(self, lower=None, upper=None):
        """
        All values less than lower will be set to lower
        All values greater than upper will be set to upper

        Parameters
        ----------
        lower: number or None
        upper: number or None

        Returns
        -------
        A DataFrame
        """
        return self._non_agg(np.clip, a_min=lower, a_max=upper)

    def round(self, n):
        """
        Rounds values to the nearest n decimals

        Returns
        -------
        A DataFrame
        """
        return self._non_agg(np.round, decimals=n)

    def copy(self):
        """
        Copies the DataFrame

        Returns
        -------
        A DataFrame
        """
        return self._non_agg(np.copy)


    # **kwargs used for addtional arguments passed to the methods above like clip or round
    #kwarg will store it as a dictionary
    #shape of dataframe remains the same for the non aggregate functions
    def _non_agg(self, funcname, **kwargs):
        """
        Generic non-aggregation function
    
        Parameters
        ----------
        funcname: numpy function
        kwargs: extra keyword arguments for certain functions

        Returns
        -------
        A DataFrame
        """
        new_data= {}
        for col, values in self._data.item():
            if values.dtype.kind == 'O':
                new_data[col] = values.copy()
            else:
            #value is the underlying numpy array and kwargs will be dict containing the other list of parameters which will be upacked within this function call    
                new_data[col] = funcname(values, **kwargs)    
        return DataFrame[new_data]    

    def diff(self, n=1):
        """
        Take the difference between the current value and
        the nth value above it.

        Parameters
        ----------
        n: int

        Returns
        -------
        A DataFrame
        """
        def func(values):
            #if is a boolean or int we are converting it into a float
            values = values.astype('float')
            values_shifted = np.roll(values, n)
            values = values - values_shifted
            if n >= 0:
                values[:n] = np.NaN
            else:
                values[n:] = np.NaN
            return values        
        return self._non_agg(func)

    def pct_change(self, n=1):
        """
        Take the percentage difference between the current value and
        the nth value above it.

        Parameters
        ----------
        n: int

        Returns
        -------
        A DataFrame
        """
        def func(values):
            #if is a boolean or int we are converting it into a float
            values = values.astype('float')
            values_shifted = np.roll(values, n)
            values = (values - values_shifted)  / values_shifted  
            if n >= 0:
                values[:n] = np.NaN
            else:
                values[n:] = np.NaN
            return values        
        return self._non_agg(func)

    #### Arithmetic and Comparison Operators ####

    def __add__(self, other):
        return self._oper('__add__', other)

    def __radd__(self, other):
        return self._oper('__radd__', other)

    def __sub__(self, other):
        return self._oper('__sub__', other)

    def __rsub__(self, other):
        return self._oper('__rsub__', other)

    def __mul__(self, other):
        return self._oper('__mul__', other)

    def __rmul__(self, other):
        return self._oper('__rmul__', other)

    def __truediv__(self, other):
        return self._oper('__truediv__', other)

    def __rtruediv__(self, other):
        return self._oper('__rtruediv__', other)

    def __floordiv__(self, other):
        return self._oper('__floordiv__', other)

    def __rfloordiv__(self, other):
        return self._oper('__rfloordiv__', other)

    def __pow__(self, other):
        return self._oper('__pow__', other)

    def __rpow__(self, other):
        return self._oper('__rpow__', other)

    def __gt__(self, other):
        return self._oper('__gt__', other)

    def __lt__(self, other):
        return self._oper('__lt__', other)

    def __ge__(self, other):
        return self._oper('__ge__', other)

    def __le__(self, other):
        return self._oper('__le__', other)

    def __ne__(self, other):
        return self._oper('__ne__', other)

    def __eq__(self, other):
        return self._oper('__eq__', other)

    def _oper(self, op, other):
        """
        Generic operator function

        Parameters
        ----------
        op: str name of special method
        other: the other object being operated on

        Returns
        -------
        A DataFrame
        """
        if isinstance(other, DataFrame):
            if other.shape[1] != 1:
                raise TypeError("dataFrame must be single column")
            else:
                other = next(iter(other._data.values()))

        new_data = {}  
        for col, values in self._data.items():
        # getattr will return the method when we provide the string, 
        # this is why op is defined as a string here
        # other will contain the 5 if there is for example - df + 5 
        # other could be anything from numpy array to int,float or bool
        # getattr means that find the underlying op method of the values attribute which is numpy array
        method = getattr(values, op)
        new_data[col] = method(other)
        return DataFrame(new_data)
        
                 
                         

    def sort_values(self, by, asc=True):
        """
        Sort the DataFrame by one or more values

        Parameters
        ----------
        by: str or list of column names
        asc: boolean of sorting order

        Returns
        -------
        A DataFrame
        """
        if isinstance(by,str):
            #order is a numpy array
            order = np.argsort(self._data[by])
        elif isinstance(by, list):
            # this will give the list of numpy arrays 
            #lexsort works in way where the array you want to sort first comes last 
            # that is why we reverse the list here
            by = [self._data[col] for col in by[::-1]] 
            order = np.lexsort(by)
        else:
            raise TypeError("By must be either a list or string")      

        if not asc:
            #this is how reverse a list or numpy array
            order = order[::-1]
 

       # we are using here the get item method which we have already implemented 
       # where we implemented selection of rows with a list in the get item
       # : means that we are selecting all of the columns, thats why the result shows all the columns,
       # and sorting is done on the columns which is defined in the argument
       # below two parameters are rows and columns - selecting all the rows
        return self[order.toList(), :]    

    def sample(self, n=None, frac=None, replace=False, seed=None):
        """
        Randomly samples rows the DataFrame

        Parameters
        ----------
        n: int
            number of rows to return
        frac: float
            Proportion of the data to sample
        replace: bool
            Whether or not to sample with replacement
        seed: int
            Seeds the random number generator

        Returns
        -------
        A DataFrame
        """
        #choices chooses randomly from a collection of objects 
        #replace values will be passed to replace parameter of the choice method
        #self is the dataframe object here
        #if the seed is given, then that seed will passed on to the seed method
        # n must be a integer or a frac
        # (range(len(self)) means selecting from the possible values of row numbers

        if seed:
            np.random.seed(seed)
        if frac:
            if frac <= 0:
                raise ValueError("Fraction must be postive")
                n = int(frac * len(self))
                if not isinstance(n, int):
                    raise TypeError("N is not an integer")

        rows = np.random.choice(range(len(self)), n, replace=replace)

        return self[row.toList, :]        


         
        

    def pivot_table(self, rows=None, columns=None, values=None, aggfunc=None):
        """
        Creates a pivot table from one or two 'grouping' columns.

        Parameters
        ----------
        rows: str of column name to group by
            Optional
        columns: str of column name to group by
            Optional
        values: str of column name to aggregate
            Required
        aggfunc: str of aggregation function

        Returns
        -------
        A DataFrame
        """
        if rows is None and columns is None:
            raise Value Error('`rows` and `values` cannot be None')


        if values is not None:
            val_data = self._data[values]
            if aggfunc is None:
                raise ValueError("Aggfunc must be provided with given values")
        else:
            if aggfunc is None:
                aggfunc = 'size'
                val_data = np.empty(len(self))
            else:
                raise ValueError("You cannot provide aggfunc when the values are none")    


        if rows is None:
            row_values = self._data[rows]   

        if columns is None:
            col_values = self._data[columns]

        # deciding the pivot type based on the input parameters of the method 
        if rows is None:
            pivot_type = 'columns'
        elif columns is None:
            pivot_type = 'rows'   
        else:
            pivot_type = 'all'

        
        from collections import defaultdict
        #below means that every key is mapped to a empty list in the dictionary
        d = defaultdict(list)   
        if pivot_type == 'columns':
            for group, val in zip(col_data, val_data):
                d[group].append(val)
        elif pivot_type == 'row':
            for group, val in zip(row_data, val_data):
                d[group].append(val)
        else:
             for group1, group2, val in zip(row_data, col_data, val_data):
                 d[(group1, group2)].append(val)       

        #Now we will map the groups with the aggregated values, above it is mapped to raw values
        #for aggregation we will convert the list into a numpy array
        #aggfunc is given as a string in this method parameters

        agg_dict = {}

        for group, value in d.items():
            arr = np.values(value)
            func= getattr(np, aggfunc)
            agg_dict[group] = func(arr)


        # Creating a new dataframe 
        # when we pass agg_dict to sorted, it will iterate through just the keys of agg dict

        new_data = {}

        if pivot_type == 'columns':
            for col in sorted(agg_dict):
                #follwing is converting the list into np array as we have seen many times in the code
                # converting the row values into columns
                new_data[col] = np.array([agg_dict[col]])
        #Keeping rows as rows, assigning aggregated row_vals 
        elif pivot_type == 'rows':
            row_vals = np.array([agg_dict.keys()])
            vals = np.array([agg_dict.values()])

            #sorted

            order = np.argsort(row_vals)

            # rows staying as rows
            new_data[rows] = row_vals[order]
            # putting aggregated vals in another aggfunc column
            new_data[aggfunc] = vals[order]
        # when both rows and columns are present
        else:
            row_set = {}
            col_set = {}
            for group in agg_dict:
                row_set.add(group(0))
                col_set.add(group(1))
            #ordering so there is no mismatch    
            row_list = sorted(row_set)  # apples, oranges
            col_list = sorted(col_set)  # texas, florida
            #keeping the rows as same    
            new_data[rows] = agg_dict[row_list] #putting apples and oranges inside fruit
            #iterating through the column parameters, and then with iterating through rows within columns
            for col in col_list:  #iterating through state
                new_vals = []
                for row in row_list: # iterating through fruits within each state
                    new_val = agg_dict.get((row, col), np.nan) #getting the values (fruit,state) pair
                    new_vals.append(new_val)

                new_data[col] = np.array(new_vals) #texas wll become a column

            return DataFrame(new_data)         




        return DataFrame(new_data)        







    def _add_docs(self):
        agg_names = ['min', 'max', 'mean', 'median', 'sum', 'var',
                     'std', 'any', 'all', 'argmax', 'argmin']
        agg_doc = \
        """
        Find the {} of each column
        
        Returns
        -------
        DataFrame
        """
        for name in agg_names:
            getattr(DataFrame, name).__doc__ = agg_doc.format(name)


# We are able to manipulate the strings in our dataframe with an entirely new class
class StringMethods:
    #the self above in the dataframe class will be passed onto the df parameter here
    def __init__(self, df):
        self._df = df

    #str method is generid method that would handle all the other methods, we have to implement it 
    def capitalize(self, col):
        return self._str_method(str.capitalize, col)

    def center(self, col, width, fillchar=None):
        if fillchar is None:
            fillchar = ' '
        return self._str_method(str.center, col, width, fillchar)

    def count(self, col, sub, start=None, stop=None):
        return self._str_method(str.count, col, sub, start, stop)

    def endswith(self, col, suffix, start=None, stop=None):
        return self._str_method(str.endswith, col, suffix, start, stop)

    def startswith(self, col, suffix, start=None, stop=None):
        return self._str_method(str.startswith, col, suffix, start, stop)

    def find(self, col, sub, start=None, stop=None):
        return self._str_method(str.find, col, sub, start, stop)

    def len(self, col):
        return self._str_method(str.__len__, col)

    def get(self, col, item):
        return self._str_method(str.__getitem__, col, item)

    def index(self, col, sub, start=None, stop=None):
        return self._str_method(str.index, col, sub, start, stop)

    def isalnum(self, col):
        return self._str_method(str.isalnum, col)

    def isalpha(self, col):
        return self._str_method(str.isalpha, col)

    def isdecimal(self, col):
        return self._str_method(str.isdecimal, col)

    def islower(self, col):
        return self._str_method(str.islower, col)

    def isnumeric(self, col):
        return self._str_method(str.isnumeric, col)

    def isspace(self, col):
        return self._str_method(str.isspace, col)

    def istitle(self, col):
        return self._str_method(str.istitle, col)

    def isupper(self, col):
        return self._str_method(str.isupper, col)

    def lstrip(self, col, chars):
        return self._str_method(str.lstrip, col, chars)

    def rstrip(self, col, chars):
        return self._str_method(str.rstrip, col, chars)

    def strip(self, col, chars):
        return self._str_method(str.strip, col, chars)

    def replace(self, col, old, new, count=None):
        if count is None:
            count = -1
        return self._str_method(str.replace, col, old, new, count)

    def swapcase(self, col):
        return self._str_method(str.swapcase, col)

    def title(self, col):
        return self._str_method(str.title, col)

    
    def lower(self, col):
        return self._str_method(str.lower, col)

    def upper(self, col):
        return self._str_method(str.upper, col)

    def zfill(self, col, width):
        return self._str_method(str.zfill, col, width)

    def encode(self, col, encoding='utf-8', errors='strict'):
        return self._str_method(str.encode, col, encoding, errors)


    # Anything with a two star argument will take the extra arguments as keywords
    # Anything with 1 star takes extra arguments as tuples 
    # self is the current instance of string methods here
    def _str_method(self, method, col, *args):
        value = self._df._data[col]        
        if value.dtype.kind = 'O':  
            raise TypeError("Str methods only work with columns having strings")
        #value is array of string and val is a single string here
        new_vals = []
        for val in value:
            if val is None:
                new_vals.append(None)
            new_vals.append(method(val, *args))

        return DataFrame(col: np.array(new_vals))    

#read_csv is function and not a method, it is defined on module level
#this function is not bound to any object
def read_csv(fn):
    """
    Read in a comma-separated value file as a DataFrame

    Parameters
    ----------
    fn: string of file location

    Returns
    -------
    A DataFrame
    """
    #this can be imported at the top as well
    from collections import defaultdict
    data = defaultdict(list)
    #fn is the file name
    with open(fn) as file:
        header = file.readline()
        column_header = header.strip('\n').strip(',') #this will output an array of column headers
        # files are iterable, so iterating through the file to get the rest of the column values
        for line in file:
            values = line.strip('\n').split(',')
            for col, values in zip(column_header, values):
                data[col].append(values) 

    new_data= {}
    #col is string and vals is a list of strings
    #we need to convert the list of strings into a correct datatype
    # we try to convert the list of string int or float, if it does not work we keep it as a list of strings
    for col, vals in data.items():
        try: 
            new_data[col] = np.array(vals, dtype = 'int')
        except ValueError:
            try:
                new_data[col] = np.array(vals, dtype = 'float')
            except:
                new_data[col] = np.array(vals, dtype = 'object')    

    return DataFrame(new_data)                











