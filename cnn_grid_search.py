'''
Cross-validated grid search for CNN training
'''


# Author: Filippo Arcadu
#         09/05/2018




from __future__ import division , print_function
import argparse
import sys , os
import yaml
import numpy as np
import time
import random
import itertools

from sklearn.model_selection import StratifiedKFold

import pandas as pd

import multiprocessing as mproc




# =============================================================================
# Types 
# =============================================================================

myfloat = np.float32
myint   = np.int




# =============================================================================
# List of available cross-validation techniques 
# =============================================================================

list_available_cv = [ 'random-resampling' , 'k-fold' ]




# =============================================================================
# Image formats 
# =============================================================================

IMG_EXT = ( '.tif' , '.tiff' , '.png' , '.jpg' , '.jpeg' , '.npy' )



# =============================================================================
# CSV separator 
# =============================================================================

SEP = ','



# =============================================================================
# Parse inpuy arguments
# =============================================================================

def _examples():
    print( "\n\nEXAMPLE\n\nDo 5-fold cross-validation with patient-id constraint:\n\n" \
            "'''\npython cnn_grid_search.py -i /pstore/data/pio/Tasks/PIO-451_exp-Delphi-MICCAI/cnn_all_fields_drprog/data/ride_rise_dr-prog_month06.csv -o /pstore/data/pio/Tasks/PIO-451_exp-Delphi-MICCAI/cnn_all_fields_drprog/dl/grid_search/month06/ -p /pstore/data/pio/Tasks/PIO-451_exp-Delphi-MICCAI/cnn_all_fields_drprog/dl/configs/cnn_config.yml -col-file 'Filepath Crop' -col-label delta_drss_month06_binary -col-constr PATNUM --path-logs /pstore/data/pio/Tasks/PIO-451_exp-Delphi-MICCAI/cnn_all_fields_drprog/dl/logs/ -n 5 --qos long --myenv /pstore/data/pio/Tasks/PIO-40/myenv_latest_keras/bin/activate -c k-fold -l m06-all -r1 'optimizer:param_adam:learning_rate;optimizer_fine_tuning:param_adam:learning_rate' -r2 '1e-5,5e-5,1e-4,5e-4,1e-3,5e-3;1e-5,5e-5,1e-4,5e-4,1e-3,5e-3'\n'''\n\n" )    
          


def _get_args():
    parser = argparse.ArgumentParser(
                                        prog='cnn_grid_search',
                                        description='Cross-validated grid-search',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                        add_help=False
                                     )
    
    parser.add_argument('-i', '--csv_input', dest='csv_input',
                        help='Input training CSV')                                            
    
    parser.add_argument('-o', '--path_out', dest='path_out',
                        help='Output path')                        

    parser.add_argument('-p', '--file_config', dest='file_config',
                        help='Input YAML configuration file to be modified. Passed as is if --param_list and --value_list are not set.')                        
    
    parser.add_argument('-c', '--cv-type', dest='cv_type', default='k-fold',
                        help='Specify which type of cross-validation technique you want to use ' \
                             + ' "random-resampling" or "k-fold"')
                        
    parser.add_argument('-n', '--num_folds', dest='num_folds', type=myint, default=10,
                        help='Number of cross-validation folds or random repetitions')

    parser.add_argument('--perc_valid', dest='perc_valid', type=myint, default=20,
                        help='perc_validentage of dataset for validation when dealing with random repetitions')
                       
    parser.add_argument('-col-constr', dest='col_constr',
                        help='Select name of column containing splitting constraint' )

    parser.add_argument('-col-file', dest='col_file',
                        help='Select name of column containing image filenames' )

    parser.add_argument('-col-label', dest='col_label',
                        help='Select name of column containing labels' )                        
    
    parser.add_argument('-r1', '--param_list', dest='param_list',
                        help='OPTIONAL' + \
                             'Specify parameters belonging to YAML to be explored; multiple parameters must be separated by the character ";"' + \
                             'each parameter should be specified with characters ":" to identify the dictionary hierarchy, e.g.:' + \
                             ' -p optimizer_transfer_learning:param_sgd:_learning_rate')
 
    parser.add_argument('-r2', '--value_list', dest='value_list',
                        help='OPTIONAL' + \
                             'Specify range in which parameters have to be span; the range of a single parameter must be indicated either in ' + \
                             'the form min_value:max_value:num_values or value1,value2,value3,value4; ranges corresponding to multiple parameters ' + \
                             'should be separated by the character ";"')

    parser.add_argument('-l', '--label', dest='label', 
                        help='Specify label characterizing the batch of jobs to execute')

    parser.add_argument('-e', '--email', dest='email',
                        help='Specify email where to send the outcome of the Slurm jobs')

    parser.add_argument('--path-logs', dest='path_logs',
                        help='Specify where to save the output Slurm files')    
    
    parser.add_argument('--qos', dest='qos', default='normal', 
                        help='Specify where to save the output Slurm files') 

    parser.add_argument('--myenv', dest='myenv', 
                        help='Specify path to virtualenv') 

    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='Enable debugging mode, i.e. see what command lines are ' + \
                             'created without executing them')                        

    parser.add_argument('-h', '--help', dest='help', action='store_true',
                        help='Print help and examples')                            

    args = parser.parse_args()
    
    if args.help is True:
        parser.print_help()
        _examples()
        sys.exit()

    if args.csv_input is None:
        parser.print_help()
        sys.exit('\nERROR ( _get_args ): Input CSV has not been specified!\n')
        
    if os.path.isfile( args.csv_input ) is False:
        parser.print_help()
        sys.exit('\nERROR ( _get_args ): Input training CSV ' + args.csv_input + ' does not exist!\n')
        
    if args.path_out is None:
        parser.print_help()
        sys.exit('\nERROR ( _get_args ): Output path has not been specified!\n') 

    if os.path.isdir( args.path_out ):
        parser.print_help()
        sys.exit('\nERROR ( _get_args ): Output path ' + args.path_out + ' already exists, cannot overwrite!\n')         

    if args.file_config is None:
        parser.print_help()
        sys.exit('\nERROR ( _get_args ): Input configuration YAML file has not been specified!\n')
        
    if os.path.isfile( args.file_config ) is False:
        parser.print_help()
        sys.exit('\nERROR ( _get_args ): Input configuration YAML does not exist!\n')   
        
    if args.col_file is None:
        parser.print_help()
        sys.exit('\nERROR ( _get_args ): Name of column containing the image filenames has not been specified!\n')                

    if args.col_label is None:
        parser.print_help()
        sys.exit('\nERROR ( _get_args ): Name of column containing the labels has not been specified!\n')  

    if args.cv_type not in list_available_cv:
        parser.print_help()
        sys.exit('\nERROR ( _get_args ): The selected type of cross-validation ' + args.cv_type  + \
                 ' is not available!\nPlease select among ' + ','.join( list_available_cv ) + '\n')  

    return args
                     
                    
                    
                     
# =============================================================================
# Definition of class GridSearch
# =============================================================================

class GridSearch:
    
    # ===================================
    # Init 
    # ===================================

    def __init__( self                             , 
                  csv_input                        ,
                  path_out                         , 
                  file_config                      ,
                  col_file                         ,
                  col_label                        ,
                  param_list                       , 
                  value_list                       ,                
                  col_constr = None                ,
                  cv_type    = 'random-resampling' ,
                  num_folds  = 10                  ,
                  perc_valid = 20                  ,
                  path_logs  = None                , 
                  qos        = 'normal'            ,
                  myenv      = None                ,
                  debug      = False               ):
        
        # Assign inputs to class attributes
        self._csv_input  = csv_input
        self._path_out   = self._create_path( path_out )
        self._col_file   = col_file
        self._col_label  = col_label
        self._col_constr = col_constr
        self._cv_type    = cv_type
        self._perc_valid = perc_valid
        self._qos        = qos
        self._myenv      = myenv
        self._debug      = debug
        
        if self._cv_type not in list_available_cv:
            sys.exit( '\nERROR ( CrossVal -- init ): ' + self._cv_type + \
                      ' is not available!\nChoose among '.join( list_available_cv , ',' ) )
        
        if path_logs is None:
            self._path_logs = path_logs
        else:
            self._path_logs = self._create_path( path_logs )
            
        self._num_folds  = num_folds
            
            
        # Read config file
        self._file_config = file_config
        self._read_yml()
            
         
        # Get parameters
        self._get_parameters( param_list )
        

        # Check that input parameters are available
        # inside config file
        self._check_params()

        
        # Get parameter values
        self._get_input_values( value_list )


        # Create tables
        self._create_tables()


        # Create output directories
        self._create_output_folders()

           
        
    # ===================================
    # Create path if it does not exist alredy
    # ===================================

    def _create_path( self , path ):
        if os.path.isdir( path ) is False:
            os.mkdir( path )
        return path

            
            
    # ===================================
    # Rea YAML file
    # ===================================

    def _read_yml( self ):
        with open( self._file_config , 'r' ) as ymlfile:
            cfg = yaml.load( ymlfile )
        
        self._config = cfg



    # ===================================
    # Write YAML file
    # ===================================

    def _write_yaml( self , filename , config ):
        with open( filename , 'w' ) as outfile:
            yaml.dump( config , outfile , default_flow_style=False )
        


    # ===================================
    # Get list of input parameters
    # ===================================

    def _get_parameters( self , param_string ):
        # Initialize list
        self._param_list = []

        if param_string is not None:
            if param_string.find( ';' ) != -1:
                chunks = param_string.split( ';' )

                for i in range( len( chunks ) ):
                    self._param_list.append( chunks[i].split( ':' ) )

            # Case 2: single parameters
            else:
                self._param_list.append( param_string.split( ':' ) )



    # ===================================
    # Check existence of parameters inside YAML
    # ===================================

    def _check_params( self ):
        for i in range( len( self._param_list ) ):
            nn = len( self._param_list[i] )
        
            if nn == 2:
                try:
                    entry = self._config[self._param_list[i][0]][self._param_list[i][1]]                
                except:
                    sys.exit( "\nERROR: YAML file has no variable callable as config['" + \
                              self._param_list[i][0] + "']['" + self._param_list[i][1] + "']\n")

            elif nn == 3:
                try:
                    entry = self._config[self._param_list[i][0]][self._param_list[i][1]][self._param_list[i][2]]
                except:
                    sys.exit( "\nERROR: YAML file has no variable callable as config['" + \
                              self._param_list[i][0] + "']['" + self._param_list[i][1] + \
                              "']['" + self._param_list[i][2] + "']\n")
        


    # ===================================
    # Identify whether string is number
    # and which format
    # ===================================

    def _identify_number_string( self , string ):
        # Try to cast 1st value as float        
        try:
            value  = myfloat( string )
            type_value = 'float'
            return type_value            
        except:
            pass

        # Try to cast 2nd value as int
        try:
            value  = myint( string )
            type_value = 'int'
            return type_value
        except: 
            pass
    


    # ===================================
    # Get range of values for a certain parameter
    # ===================================
    
    def _get_value_array( self , string ):
        # Case 1: range
        if string.find( ':' ) != -1:
            chunks     = string.split( ':' )
            type_value = self._identify_number_string( chunks[0] )

            # Case floating values
            if type_value == 'float':
                min_value   = myfloat( chunks[0] )
                max_value   = myfloat( chunks[1] )
                num_values  = myint( chunks[2] )
                value_array = np.linspace( min_value , max_value , num_values ).astype( myfloat )

            # Case integer values
            elif type_value == 'int':
                min_value   = myint( chunks[0] )
                max_value   = myint( chunks[1] )
                num_values  = myint( chunks[2] )
                value_array = np.linspace( min_value , max_value , num_values ).astype( myint ) 


        # Case 2: list of values:
        elif string.find( ',' ) != -1:
            chunks     = string.split( ',' )
            type_value = self._identify_number_string( chunks[0] )

            # Case floating values:
            if type_value == 'float':
                value_array = np.array( chunks ).astype( myfloat )

            # Case integer values:
            elif type_value == 'int':
                value_array = np.array( chunks ).astype( myint ) 

            #  Case integer values:
            else:
                value_array = np.array( chunks )


        # Case 3: just 1 value
        else:
            type_value = self._identify_number_string( string )

            # Case floating values:
            if type_value == 'float':
                value_array = np.array( [ string ] ).astype( myfloat )

            # Case integer values:
            elif type_value == 'int':
                value_array = np.array( [ string ] ).astype( myint ) 

            #  Case integer values:
            else:
                value_array = np.array( [ string ] )

        return value_array



    # ===================================
    # Get input values
    # ===================================
    
    def _get_input_values( self , value_string ):
        # Initialize list
        self._value_list = []

        if value_string is not None:
            # Case 1: multiple ranges
            if value_string.find( ';' ) != -1:
                chunks = value_string.split( ';' )

                for i in range( len( chunks ) ):
                    self._value_list.append( self._get_value_array( chunks[i] ) )

            # Case 2: single parameters
            else:   
                self._value_list.append( self._get_value_array( value_string ) )

        
        # Get number of tuples
        if len( self._value_list ) == 0:
            self._tuples = None
        elif len( self._value_list ) == 1:
            self._tuples = self._value_list
        elif len( self._value_list ) == 2:
            self._tuples = list( itertools.product( self._value_list[0] , 
                                              self._value_list[1] ) )
        elif len( self._value_list ) == 3:
            self._tuples = list( itertools.product( self._value_list[0] , 
                                              self._value_list[1] , 
                                              self._value_list[2] ) )
        elif len( self._value_list ) == 4:
            self._tuples = list( itertools.product( self._value_list[0] , 
                                              self._value_list[1] , 
                                              self._value_list[2] ,
                                              self._value_list[3] ) )        



    # ===================================
    # Create CSV tables 
    # ===================================

    def _create_tables( self ):
        # Read training and validation CSV
        self._df = pd.read_csv( self._csv_input , sep=SEP )
   

        # Create CSV folder
        self._path_csv = self._create_path( os.path.join( self._path_out , 'csv/' ) )
        self._path_cfg = self._create_path( os.path.join( self._path_out , 'configs/' ) )
        
        
        # Column according which to split the dataset
        col_file0 = self._col_file.split( ':' )[0]

        if self._col_constr is None:
            col_split = col_file0
        else:
            col_split = self._col_constr
            
        self._ref_all = self._df[ col_split ].values
        
        
        # Get unique indeces
        self._ref_unique , self._ind_unique = np.unique( self._ref_all , 
                                                         return_index=True )
        
 
        # Get split ratio
        if self._cv_type == 'random-resampling':
            self._nval = myint( len( self._ref_unique ) * self._perc_valid / 100.0 )
        elif self._cv_type == 'k-fold':
            self._nval = myint( len( self._ref_unique ) / ( 1.0 * self._num_folds ) )
        
            
        # Initialize lists
        self._csv_train_list = []
        self._csv_valid_list = []
        self._dir_outputs    = []
       

        # Case A: random resampling CV
        if self._cv_type == 'random-resampling':
            string_cv_type = 'rr'
            df_t_list      = []
            df_v_list      = []
            
            for i in range( self._num_folds ):
                df_t , df_v    = self._split_dataset_random_resampling( self._df )
                df_t_list.append( df_t )
                df_v_list.append( df_v )


        # Case B: k-fold CV
        elif self._cv_type == 'k-fold':
            df_t_list , df_v_list = self._split_dataset_kfold( self._df )
            string_cv_type = 'kf'
            

        # Split dataset
        for i in range( len( df_t_list ) ):
            df_t = df_t_list[i]
            df_v = df_v_list[i]

            print( '\nChecking if training and validation data frames n.', i,' are disjoint with respect to ', col_file0 )
            self._is_split_disjoint( df_t[ col_file0 ] , 
                                     df_v[ col_file0 ] ,
                                     level = col_file0 )
                                     
            if self._col_constr is not None:
                print( 'Checking if training and validation data frames n.', i,' are disjoint with respect to ', self._col_constr )
                self._is_split_disjoint( df_t[ self._col_constr ] , 
                                         df_v[ self._col_constr ] ,
                                         level = self._col_constr )                                     
            if i < 10:
                num_string = '0' + str( i )
            else:
                num_string = str( i )
            
            file_train = os.path.join( self._path_csv , 'csv_train_' + string_cv_type + num_string + '.csv' )
            self._csv_train_list.append( file_train )
            df_t.to_csv( file_train , sep=SEP , index=False )
            
            file_valid = os.path.join( self._path_csv , 'csv_valid_' + string_cv_type + num_string + '.csv' )
            self._csv_valid_list.append( file_valid )
            df_v.to_csv( file_valid , sep=SEP , index=False )
            
            outcome_train = df_t[ self._col_label ].values
            outcome_valid = df_v[ self._col_label ].values

            values = np.unique( outcome_train )

            balance_train = len( outcome_train[ outcome_train == values[0] ] ) / myfloat( len( outcome_train ) ) * 100.0
            balance_valid = len( outcome_valid[ outcome_valid == values[0] ] ) / myfloat( len( outcome_valid ) ) * 100.0
            
            print( '\nFold n.', i )
            print( 'Training CSV: ', file_train )
            print( 'Validation CSV: ', file_valid )
            print( 'Training balance --> Class 0: ', balance_train,' -- Class 1: ', 100 - balance_train  )
            print( 'Validation balance --> Class 0: ', balance_valid,' -- Class 1: ', 100 - balance_valid  )


        # Case "k-fold": check that all folds do not intersect
        if self._cv_type == 'k-fold':
            self._are_folds_disjoint()
            
            
               
    # ===================================
    # Split dataset for random resampling CV
    # ===================================
    
    def _split_dataset_random_resampling( self , df_all ):     
        # Split unique indeces
        i_train , i_valid = self._split_array_randomly( np.arange( len( self._ind_unique ) ) ,
                                                        self._nval )
                                                        
                                                        
        # Split data frame
        ref_unq_train = self._ref_all[ self._ind_unique[ i_train ] ]
        ref_unq_valid = self._ref_all[ self._ind_unique[ i_valid ] ]
        
        ind_train = []
        for i in range( len( ref_unq_train ) ):
            inds = np.argwhere( self._ref_all == ref_unq_train[i] ).reshape( -1 )
            for ind in inds:
                ind_train.append( ind )
                
        ind_valid = []
        for i in range( len( ref_unq_valid ) ):
            inds = np.argwhere( self._ref_all == ref_unq_valid[i] ).reshape( -1 )
            for ind in inds:
                ind_valid.append( ind )
                
        df_t = df_all.iloc[ ind_train ]
        df_v = df_all.iloc[ ind_valid ]
        
        return df_t , df_v
                
        
        
    # ===================================
    # Split dataset for k-fold CV
    # ===================================
    
    def _split_dataset_kfold( self , df_all ):
        # Initialize lists
        df_t_list = []
        df_v_list = []


        # Initialize stratified k-fold split
        skf = StratifiedKFold( n_splits = self._num_folds )


        # Split unique indeces
        ind_span   = np.arange( len( self._ind_unique ) )
        outcome_un = df_all[ self._col_label ].values[ self._ind_unique ]


        # For loop k-fold splitting
        for i_train , i_valid  in skf.split( ind_span , outcome_un ):
            ref_unq_train = self._ref_all[ self._ind_unique[ i_train ] ]
            ref_unq_valid = self._ref_all[ self._ind_unique[ i_valid ] ]
        
            ind_train = []
            for i in range( len( ref_unq_train ) ):
                inds = np.argwhere( self._ref_all == ref_unq_train[i] ).reshape( -1 )
                for ind in inds:
                    ind_train.append( ind )
                
            ind_valid = []
            for i in range( len( ref_unq_valid ) ):
                inds = np.argwhere( self._ref_all == ref_unq_valid[i] ).reshape( -1 )
                for ind in inds:
                    ind_valid.append( ind )
                
            df_t = df_all.iloc[ ind_train ]
            df_v = df_all.iloc[ ind_valid ]

            df_t_list.append( df_t )
            df_v_list.append( df_v )
        
        return df_t_list , df_v_list    
    
    
    
    # ===================================
    # Split array randomly 
    # ===================================
    
    def _split_array_randomly( self , ii , nval ):
        # Create array of validation indeces
        i_valid   = random.sample( ii , nval )
        
        # Difference array
        i_train   = np.setdiff1d( ii , i_valid )
        
        return i_train , i_valid
        
        
    
    # ===================================
    # Check whether the two sets are disjoint 
    # ===================================
    
    def _is_split_disjoint( self , arr1 , arr2 , level='filepaths' ):      
        arr1 = np.array( arr1 )
        arr2 = np.array( arr2 )
    
        if len( np.intersect1d( arr1 , arr2 ) ):
            sys.exit( '\nERROR ( KfoldCrossVal -- _is_split_disjoint ): array1 ' + \
                      'and array2 are not disjoint at the level of ' + level + '!\n' )   
        


    # ===================================
    # Check whether folds are disjoint 
    # ===================================
 
    def _are_folds_disjoint( self ):
        import itertools
        combs = list( itertools.combinations( self._csv_valid_list , 2 ) )
        
        for i in range( len( combs ) ):
            file1 = combs[i][0]
            file2 = combs[i][1]
        
            df1 = pd.read_csv( file1 , sep=SEP )
            df2 = pd.read_csv( file2 , sep=SEP )

            col_file0 = self._col_file.split( ':' )[0]
           
            print( '\nChecking if <', file1,'> and <', file2,'> are disjoint with respect to ', col_file0 )
            self._is_split_disjoint( df1[ col_file0 ].values ,
                                     df2[ col_file0 ].values ,
                                     level=col_file0 )
      
            if self._col_constr is not None:
                print( 'Checking if <', file1,'> and <', file2,'> are disjoint with respect to ', self._col_constr )
                self._is_split_disjoint( df1[ self._col_constr ].values ,
                                         df2[ self._col_constr ].values ,
                                         level=self._col_constr )

    
                
    # ===================================
    # Create output folders
    # ===================================

    def _create_output_folders( self ):
        # Create "mother" output folder
        self._path_mod = self._create_path( os.path.join( self._path_out , 'outputs/' ) )

        
        # Create grid search output folders
        self._dir_outputs = []
        
        for i in range( len( self._tuples ) ):
            if i < 10:
                str_tuple = '00' + str( i )
            elif i < 100:
                str_tuple = '0' + str( i )
            else:
                str_tuple = str( i )
            str_tuple = 'grid_point_' + str_tuple

            self._path_grid = self._create_path( os.path.join( self._path_mod , 'outputs_' + str_tuple ) )

            for j in range( self._num_folds ):
                if j < 10:
                    str_cv = '00' + str( j )
                elif j < 100:
                    str_cv = '0' + str( j )
                else:
                    str_cv = str( j )
                
                if self._cv_type == 'k-fold':
                    str_cv = 'kf' + str_cv
                elif self._cv_type == 'random-resampling':
                    str_cv = 'rs' + str_cv
                
                output_tmp = self._create_path( os.path.join( self._path_grid , 'outputs_' + str_cv ) )
                self._dir_outputs.append( output_tmp )
 


    # ===================================
    # Create base command line for Slurm 
    # ===================================

    def _create_slurm_command( self , label=None , email=None ):
        # Specify Slurm options
        if self._qos == 'short':
            options = '--qos SHORT --time 00-03:00:00 --cpus-per-task 24 '
        elif self._qos == 'normal':
            options = '--qos normal --time 03-00:00:00 --cpus-per-task 24 '
        elif self._qos == 'long':
            options = '--qos long --time 15-00:00:00 --cpus-per-task 24 '
        
        if label is not None:
            options += ' --job-name ' + label + ' '
    
        if email is not None:
            options += ' --mail-type=END,FAIL --mail-user=' + email + ' '
        
        
        # Load modules and activate virtualenv
        if self._myenv is None:
            options += '--wrap="ml purge;ml python/python3.6-2018.05;' + \
                       'ml TensorFlow/1.12.0-foss-2018b-Python-3.6.5-2018.05;' + \
                       'export KMP_BLOCKTIME=0;' + \
                       'export MKL_NUM_THREADS=1;' + \
                       'export OMP_NUM_THREADS=1;' + \
                       'source /pstore/data/pio/ShellTools/VENVS/CPU/venv-3.6-cpu/bin/activate;'
        else:
             options += '--wrap="ml python3;ml virtualenv;' + \
                       'source ' + self._myenv + ';'
                   

        # Absolute path
        path = os.path.abspath( os.getcwd() )
        
        
        # Add sbatch command
        command = 'sbatch ' + options + 'python ' + os.path.join( path , 'cnn_train.py' ) 


        # Assign to class
        self._command = command
    


    # ===================================
    # Construct auxiliary YAML files and send Slurm jobs
    # ===================================

    def _hash_config_file( self ):
        unique_hash  = str( random.getrandbits( 128 ) )
        basename , _ = os.path.splitext( os.path.basename( self._file_config ) )
        file_hash    = os.path.join( self._path_cfg , 
                                     basename + '_' + str( unique_hash ) + '.yml' )
        return file_hash

    
    
    def _run_jobs( self ):
        # Debugging mode if enabled
        if self._debug:
            print( '\nRunning in debugging mode: no job sent to Slurm' )                                        

        # Change directory if enabled
        if self._path_logs is not None:
            print( '\nChanged running directory to ' , self._path_logs )
            os.chdir( self._path_logs )
    
    
        # Start loop
        if self._tuples is None:
            for i in range( self._num_folds ):          
                # Create hashed filename for config file
                file_yaml = self._hash_config_file()
       
                # Write new hashed YAML file
                if self._debug is False:
                    self._write_yaml( file_yaml , self._config )
                
                # Kfold string
                if i < 10:
                    string_fold = '0' + str( i )
                else:
                    string_fold = str( i )
            
                # Assemble command
                dir_output = self._dir_outputs[i]
                                       
                command    = self._command + ' -o ' + dir_output + ' -i1 ' + self._csv_train_list[i] + \
                                ' -i2 ' + self._csv_valid_list[i] + ' -col-label ' + self._col_label + \
                                ' -col-imgs ' + self._col_file + ' -l ' + self._cv_type + string_fold + \
                                ' -p ' + file_yaml + '"'
                print( '\n', command )
            
                if self._debug is False:
                    os.system( command )

        else:
            k = 0
            for i1 in range( len( self._tuples ) ):
                print( '\n\nGrid point n.',i1,' out of ', len( self._tuples ) )
                print( 'Params --> ', self._param_list )
                print( 'Tuple --> ', self._tuples[i1] )

                for i2 in range( self._num_folds ):          
                    
                    for i3 in range( len( self._tuples[i1] ) ):
                        # Modify config file
                        nn = len( self._param_list[i3] )

                        if nn == 2:
                            self._config[self._param_list[i3][0]][self._param_list[i3][1]] = str( self._tuples[i1][i3] )
                        elif nn == 3:
                            self._config[self._param_list[i3][0]][self._param_list[i3][1]][self._param_list[i3][2]] = str( self._tuples[i1][i3] )
                        elif nn == 4:
                            self._config[self._param_list[i3][0]][self._param_list[i3][1]][self._param_list[i3][2]][self._param_list[i3][3]] = str( self._tuples[i1][i3] )

                    # Create hashed filename for config file
                    file_yaml = self._hash_config_file()
       
                    # Write new hashed YAML file
                    self._write_yaml( file_yaml , self._config )
                
                    # Kfold string
                    if i2 < 10:
                        string_fold = '0' + str( i2 )
                    else:
                        string_fold = str( i2 )
            
                    # Assemble command
                    dir_output = self._dir_outputs[k]
                    k         += 1
                                       
                    command    = self._command + ' -o ' + dir_output + ' -i1 ' + self._csv_train_list[i2] + \
                                    ' -i2 ' + self._csv_valid_list[i2] + ' -col-label ' + self._col_label + \
                                    ' -col-imgs ' + self._col_file + ' -l ' + self._cv_type + string_fold + \
                                    ' -p ' + file_yaml + '"'
                    print( '\n', command )
            
                    if self._debug is False:
                        os.system( command )




# =============================================================================
# Main
# =============================================================================  

def main():
    # Get input arguments
    args = _get_args()



    # Debug mode if enabled
    if args.debug:
        print( '\n' )
        print( '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!' )
        print( '                D E B U G  M O D E              ' )
        print( '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!' )
        print( '\n' )



    # Initial print
    print( '\n\nCROSS-VALIDATED GRID SEARCH FOR CNN TRAINING\n' )
    
    
    # Initialize class
    cv = GridSearch( args.csv_input               , 
                     args.path_out                , 
                     args.file_config             ,
                     args.col_file                ,
                     args.col_label               ,
                     args.param_list              ,
                     args.value_list              ,
                     col_constr = args.col_constr ,
                     cv_type    = args.cv_type    ,
                     num_folds  = args.num_folds  ,
                     perc_valid = args.perc_valid ,
                     path_logs  = args.path_logs  , 
                     qos        = args.qos        ,
                     myenv      = args.myenv      ,
                     debug      = args.debug      )
    
    
    # Some prints    
    print( '\nType of cross-validation: ', cv._cv_type )
    print( 'Input CSV: ', cv._csv_input )
    print( 'Config file: ', cv._file_config )
    print( 'File column: ', cv._col_file )
    print( 'Label column: ', cv._col_label )    
    print( 'Output path: ', cv._path_out )

    if cv._cv_type == 'k-fold':
        print( 'Number of folds: ', cv._num_folds )
    elif cv._cv_type == 'random-resampling':
        print( 'Number of random repetitions: ', cv._num_folds )
        print( 'perc_validentage of dataset for validation: ', cv._perc_valid,'%' )
    
    print( 'QOS: ', cv._qos )
    
    if cv._path_logs is not None:
        print( '\nPath where to save slurm logfiles: ', cv._path_logs )     

    if cv._myenv is not None:
        print( '\nVirtualenv to use: ', cv._myenv )     

   
    # Create skeleton Slurm command line
    cv._create_slurm_command( label = args.label ,
                              email = args.email  )
    print( '\nSkeleton Slurm command line:\n', cv._command )
    
    
    # Create temporary YAML files
    if args.debug:
        print( '\n------- DEBUGGING MODE - NO JOB WILL BE EXECUTED -------' )
    else:
        print( '\nCreating temporary YAML files and dispatch Slurm jobs ....' )
    cv._run_jobs()
    
    if args.debug:
        print( '\n------- DEBUGGING MODE - NO JOB WILL BE EXECUTED -------' )
    
    print( '\n\n' )
    
    
    # Debug mode if enabled
    if args.debug:
        print( '\n' )
        print( '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!' )
        print( '                D E B U G  M O D E              ' )
        print( '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!' )
        print( '\n' )    
 


              
# =============================================================================
# Call to main
# =============================================================================

if __name__ == '__main__':
    main()
