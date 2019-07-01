'''
Split dataset table into training, validation and testing
according to certain constraints, if these are enabled
'''


# The routine can work in the following modes:
#
# "normal"    ---> simple random split;
# "patient"   ---> random split forcing all images with the same patient-id to be either in the training or validation set;
# "eye"       ---> random split forcing all images of the same eye to be either in the training or validation set;
# "eye-visit" ---> random split forcing all images of the same eye at the same visit to be either in the training or validation set.
# 
# The modes "patient", "eye" and "eye-visit" are based on the naming convention of RIDE and RISE, i.e.:
# given a color fundus image basename like: CF-50201-2007-10-24-M3-LE-F2-RS.png
# patient mode   ---> get "50201"
# eye mode       ---> get "50201" + "LE"
# eye-visit mode ---> all the images "CF-50201-2007-10-24-M3-LE-F2-RS" , "CF-50201-2007-10-24-M3-LE-F2-RS-1", "CF-50201-2007-10-24-M3-LE-F2-RS-2", etc ...


# Author: Filippo Arcadu 
#         AIM4EYE Project
#         11/01/2018



from __future__ import division , print_function
import argparse
import sys
import os
import time
import numpy as np
import random
import pandas as pd




# =============================================================================
# CSV separator 
# =============================================================================

SEP = ','



# =============================================================================
# Types 
# =============================================================================

myfloat = np.float32
myint   = np.int



# =============================================================================
# Constrained splitting modes 
# =============================================================================

SPLIT_MODE = [ 'normal' , 'patient' , 'eye' , 'eye-visit' ]




# =============================================================================
# Balance modes 
# =============================================================================

BALANCE_MODE = [ 'same' , 'random' , 'equal' ]





# =============================================================================
# Parsing input arguments
# =============================================================================

def _examples():
    print( '\n\nEXAMPLES\n\nSplit cross-referenced data table into a training and validation table for CNN training:\n"""\n' \
            'python cnn_split_table.py -i /pstore/data/pio/Tasks/PIO-355/original_tables/table_retouch_dataset.csv -o /pstore/data/pio/Tasks/PIO-355/dl_classification/split_tables/ -c bscan:irf -l irf_patient --constr name\n"""\n\n'
          )    

          
          
def _get_args():
    parser = argparse.ArgumentParser(    
                                        prog            = 'cnn_split_table' ,
                                        description     = 'Split cross-referenced data table' ,
                                        formatter_class = argparse.ArgumentDefaultsHelpFormatter ,
                                        add_help        = False
                                    )
    
    parser.add_argument('-i', '--table_in', dest='table_in',
                        help='Specify input CSV cross-referenced table filename')                        

    parser.add_argument('-o', '--path_out', dest='path_out', 
                        help='Specify path where to save output CSV tables')  
                        
    parser.add_argument('-t', '--test_perc', dest='test_perc', type=myint , default=10,
                        help='Specify percentage of dataset for testing (tuning)')
                        
    parser.add_argument('-v', '--valid_perc', dest='valid_perc', type=myint , default=10,
                        help='Specify percentage of dataset for validation (external)')
                        
    parser.add_argument('-c', '--columns', dest='columns', 
                        help='Specify column names of the table with ' \
                             'file reference (basename, filepath, ...) and labels, ' \
                             'example: -c fname:LETTERS' )

    parser.add_argument('-l', '--add_label', dest='add_label',
                        help='Select additional label to output files to facilitate their identification' )    

    parser.add_argument('--constr', dest='constr',
                        help='Select column constraining the random splitting')

    parser.add_argument('-b', '--balance', dest='balance', default='same',
                        help='Select class balance mode: "random", "same", "equal". ' + \
                             '"random": the class balance is not considered for the splitting; ' + \
                             '"same": keep the original class balance in all splits; ' + \
                             '"50-50": force validation and test dataset to have equal splitting for all classes')                             
                             
    parser.add_argument('-h', '--help', dest='help', action='store_true',
                        help='Print help and examples')                            

    args = parser.parse_args()
    
    if args.help is True:
        parser.print_help()
        _examples()
        sys.exit()

    if args.table_in is None:
        parser.print_help()
        sys.exit('\nERROR: Input CSV table not specified!\n')
        
    if os.path.isfile( args.table_in ) is None:
        parser.print_help()
        sys.exit('\nERROR: Input CSV table does not exist!\n')        

    if args.path_out is None:
        parser.print_help()
        sys.exit('\nERROR: Output path not specified!\n')
        
    if args.columns is None:
        parser.print_help()
        sys.exit('\nERROR: Table columns with filepaths and labels have not been specified!\n')

    if args.columns.find( ':' ) == -1:
        parser.print_help()
        sys.exit('\nERROR: Missing separator ":" for argument "columns"!\n')        

    if args.balance not in BALANCE_MODE:
        parser.print_help()
        sys.exit('\nERROR: Selected class balance mode "' + args.balance + \
                 '" is not available. Choose between ' + ' '.join( BALANCE_MODE ) + ' !' )                 
        
    return args
    
    
    

# =============================================================================
# Definition of class SplitDataset
# ============================================================================= 

class SplitDataset:
    
    # ===================================
    # Init 
    # ===================================

    def __init__( self                 , 
                  file_csv             , 
                  columns              , 
                  constr     = None    , 
                  valid_perc = 10      ,
                  test_perc  = 10      ,
                  balance    = 'random' ):
                  
        # Assign to the class fields
        self._file_csv   = file_csv
        self._constr     = constr
        self._test_perc  = test_perc
        self._valid_perc = valid_perc
        self._balance    = balance
        
        
        # Read input CSV table
        self._read_input_table( columns )
        
        
        # Get data 
        self._get_data()
        
        
        
    # ===================================
    # Read input table
    # ===================================

    def _read_input_table( self , columns ):
        # Read table
        self._table = pd.read_csv( self._file_csv , 
                                   sep    = SEP    )
        
        if self._table.shape[1] == 1:
            self._table = pd.read_csv( self._file_csv , 
                                       sep    = ';'   )
            print( 'Warning ( SplitDataset -- _read_input_table ): using ";" as CSV separator' )
                                 
                                   
        # Assign columns to class fields
        self._col_paths  = columns.split( ':' )[0]
        self._col_labels = columns.split( ':' )[1]
        
        
        # Check columns exist
        if self._col_paths not in self._table.keys():
            sys.exit( '\nERROR ( SplitDataset -- _read_input_table ): filepath column ' + \
                      self._col_paths + ' is not a key of the input table (' + \
                      ' '.join( self._table.keys() ) + ') !\n' )
                      
        if self._col_labels not in self._table.keys():
            sys.exit( '\nERROR ( SplitDataset -- _read_input_table ): label column ' + \
                      self._col_labels + ' is not a key of the input table (' + \
                      ' '.join( self._table.keys() ) + ') !\n' )



    # ===================================
    # Get filepaths and labels
    # ===================================
    
    def _get_data( self ):            
        # Get list of filepaths and labels
        self._imgs        = self._table[ self._col_paths ]
        self._labels      = self._table[ self._col_labels ]            
        self._num_imgs    = len( self._imgs )
        self._classes     = np.unique( self._labels )
        self._num_classes = len( self._classes )
        
        
        # Check filepaths are unique
        if self._num_imgs != len( np.unique( self._imgs ) ):
            diff = self._num_imgs - len( np.unique( self._imgs ) )
            sys.exit( '\nERROR: ( SplitDataset -- _get_data ): filepaths are not all unique!\n' + \
                      'There are ' + str( diff ) + ' repeated filepaths in the input table.\n\n' )
        
        
        # Get class balance
        self._class_balance = []
        
        for i in range( self._num_classes ):
            self._class_balance.append( len( np.argwhere( self._labels == self._classes[i] ) ) * 100.0 / self._num_imgs )
            
        
                
    # ===================================
    # Split table
    # ===================================

    def _split( self ):    
        # Create disjoint lists of random indeces
        self._random_disjoint_indeces()        
    
    
        # Check that arrays of indeces are indeed disjoint
        self._are_disjoint( self._ind_train , self._ind_test )
        self._are_disjoint( self._ind_train , self._ind_valid )
        self._are_disjoint( self._ind_test , self._ind_valid )
        
        
        # Create dictionaries
        if len( self._ind_train ):
            self._table_train = self._table.iloc[ self._ind_train ]
        else:
            self._table_train = None
                
        if len( self._ind_test ):
            self._table_test = self._table.iloc[ self._ind_test ]
        else:
            self._table_test = None
                    
        if len( self._ind_valid ):
            self._table_valid = self._table.iloc[ self._ind_valid ]
        else:
            self._table_valid = None
            
            
        # Evaluate class balance
        self._evaluate_class_balance()

                              
                              
    # ===================================
    # Splitting indeces
    # ===================================

    def _random_disjoint_indeces( self ):
        # Get IDs         
        if self._constr is not None:
            list_id_all = self._table[ self._constr ].values
        else:    
            list_id_all = self._imgs.copy()
            
        self._list_id , self._ind_id = np.unique( list_id_all , return_index=True )
        self._labels_id              = self._labels[ self._ind_id ]
        
        
        # Random split of ID array
        if self._constr is None:
            self._ind_train , self._ind_test , self._ind_valid = self._split_array_randomly( np.arange( len( self._list_id ) ) )
            self._num_ids_train = len( self._ind_train )
            self._num_ids_test  = len( self._ind_test )
            self._num_ids_valid = len( self._ind_valid )
            
        else:
            if self._balance == 'random':
                ind_train , ind_test , ind_valid = self._split_array_randomly( np.arange( len( self._list_id ) ) )
            else:
                ind_train , ind_test , ind_valid = self._split_array_balance()
           
            self._num_ids_train = len( ind_train )
            self._num_ids_test  = len( ind_test )
            self._num_ids_valid = len( ind_valid )            
            
            self._ind_train = []
            for i in range( len( ind_train ) ):
                self._ind_train += np.argwhere( list_id_all == self._list_id[ind_train[i]] ).tolist()
            self._ind_train = np.array( self._ind_train ).reshape( -1 )
                
            self._ind_test = []
            for i in range( len( ind_test ) ):
                self._ind_test += np.argwhere( list_id_all == self._list_id[ind_test[i]] ).tolist()
            self._ind_test = np.array( self._ind_test ).reshape( -1 )

            self._ind_valid = []
            for i in range( len( ind_valid ) ):
                self._ind_valid += np.argwhere( list_id_all == self._list_id[ind_valid[i]] ).tolist()    
            self._ind_valid = np.array( self._ind_valid ).reshape( -1 )

    
    
    # ===================================
    # Random split of indeces
    # (disregarding class balance)
    # ===================================
    
    def _split_array_randomly( self , ii ):
        # Create array of validation indeces
        num_valid = myint( self._valid_perc * len( ii ) / 100.0 )
        i_valid   = random.sample( ii.tolist() , num_valid )
        
        # Difference array
        ii_temp   = np.setdiff1d( ii.tolist() , i_valid )
        
        
        # Create array of testing indeces
        num_test  = myint( self._test_perc * len( ii ) / 100.0 )
        i_test    = random.sample( ii_temp.tolist() , num_test )

        
        # Create array of training indeces
        i_train   = np.setdiff1d( ii_temp , i_test )
        
        return i_train , i_test , i_valid



    # ===================================
    # Splitting indeces considering class balance
    # ===================================

    def _split_array_balance( self ):
        # Create array of validation indeces
        inds_aux  = np.arange( len( self._ind_id ) )
        labels    = self._labels_id
        ind_valid = self._split_per_class( inds_aux ,
                                           self._valid_perc ,
                                           labels )
        
        
        # Create difference array
        ind_temp = np.setdiff1d( inds_aux , ind_valid )


        # Create array of testing indeces
        labels   = self._labels_id[ind_temp]
        ind_test = self._split_per_class( ind_temp ,
                                          self._test_perc ,
                                          labels )                                          

        
        # Create array of training indeces
        ind_train = np.setdiff1d( ind_temp , ind_test )
        
        return ind_train , ind_test , ind_valid
        
        
        
    def _split_per_class( self , ind_all , perc , labels ):
        num_imgs = myint( perc * len( ind_all ) / 100.0 )
        ind      = []
        
        for i in range( self._num_classes ):
            if self._balance == 'same':
                num = myint( num_imgs * self._class_balance[i] / 100.0 )
            
            elif self._balance == 'equal':
                num = myint( num_imgs * 1.0 / self._num_classes )
                
            num_all = len( ind_all[ labels == self._classes[i] ] )
            
            if num < num_all:
                ind += random.sample( ind_all[ labels == self._classes[i] ].tolist() , num )
            else:
                ind += ind_all[ labels == self._classes[i] ].tolist()
        
        return np.array( ind )
        


    # ===================================
    # Check whether arrays are disjoint
    # ===================================
    def _are_disjoint( self , ind1 , ind2 ):
        intersec = np.intersect1d( ind1 , ind2 )

        if len( intersec ):
            sys.exit( '\nERROR ( KerasTree -- _are_disjoint ): training and validation indeces not disjoint!\n\n' )
    
    
    
    # ===================================
    # Evaluate class balance
    # ===================================
    
    def _evaluate_class_balance( self ):
        # Balance of training dataset
        num_imgs_class = []        
        for i in range( self._num_classes ):
            num_imgs_class.append( self._table_train[ self._table_train[ self._col_labels ] == self._classes[i] ].shape[0] )
        self._balance_train = num_imgs_class / myfloat( self._table_train.shape[0] ) * 100
        
        
        # Balance of validation dataset
        if self._table_valid is not None:
            num_imgs_class = []        
            for i in range( self._num_classes ):
                num_imgs_class.append( self._table_valid[ self._table_valid[ self._col_labels ] == self._classes[i] ].shape[0] )
            self._balance_valid = num_imgs_class / myfloat( self._table_valid.shape[0] ) * 100


        # Balance of test dataset
        if self._table_test is not None:
            num_imgs_class = []        
            for i in range( self._num_classes ):
                num_imgs_class.append( self._table_test[ self._table_test[ self._col_labels ] == self._classes[i] ].shape[0] )
            self._balance_test = num_imgs_class / myfloat( self._table_test.shape[0] ) * 100
                
    
    
    # ===================================
    # Write output tables
    # ===================================

    def _write_tables( self , pathout , add_label=None ):
    
        # Create output filenames
        basename     = os.path.basename( self._file_csv )
        basename , _ = os.path.splitext( basename )
       
        if ' ' in self._constr:
            str_constr = self._constr.replace( ' ' , '-' )
        else:
            str_constr = self._constr

        if self._constr is not None:
            stem = os.path.join( pathout , basename + '_constr-' + str_constr )
        else:
            stem = os.path.join( pathout , basename )
    
        if add_label is not None:
            stem += '_' + add_label
        
        fileout_train = stem + '_train.csv'
        fileout_test  = stem + '_test.csv'
        fileout_valid = stem + '_valid.csv'
        
        if os.path.isfile( fileout_train ) and self._table_train is not None:
            sys.exit( '\nERROR: file ' + fileout_train + ' exists, cannot overwrite!\n\n' )
            
        if os.path.isfile( fileout_test ) and self._table_test is not None:
            sys.exit( '\nERROR: file ' + fileout_test + ' exists, cannot overwrite!\n\n' )
        
        if os.path.isfile( fileout_valid ) and self._table_valid is not None:
            sys.exit( '\nERROR: file ' + fileout_valid + ' exists, cannot overwrite!\n\n' )            
                   
    
        # Write dictionaries to CSV
        if self._table_train is not None:
            self._table_train.to_csv( fileout_train , sep=SEP , index=False )
            print( '\nWritten training table in ', fileout_train )
        
        if self._table_test is not None:
            self._table_test.to_csv( fileout_test , sep=SEP , index=False )
            print( 'Written testing table in ', fileout_test )    
    
        if self._table_valid is not None:
            self._table_valid.to_csv( fileout_valid , sep=SEP , index=False )    
            print( 'Written validation table in ', fileout_valid )
    


    
# =============================================================================
# Main
# =============================================================================

def main():
    # Get input arguments
    time1 = time.time()
    args  = _get_args()
    
    print( '\nSPLIT DATASET INTO TRAINING, TESTING AND VALIDATION SET\n' )
    
    
    # Initialize class
    sdata = SplitDataset( args.table_in                ,
                          args.columns                 ,
                          constr     = args.constr       , 
                          valid_perc = args.valid_perc , 
                          test_perc  = args.test_perc  , 
                          balance    = args.balance )
    
    
    # Prints
    print( '\nInput table: ', sdata._file_csv )
    print( 'Number of table rows: ', sdata._num_imgs )
    print( 'Splitting column: ', sdata._constr )
    print( 'Balance mode: ', sdata._balance )
    print( 'Classes: ', sdata._classes )
    print( 'Class balance (%): ', sdata._class_balance )
    print( '\n' )
    
    if sdata._test_perc:
        print( 'Testing data ratio (tuning data): ', sdata._test_perc,'%' )
    if sdata._valid_perc:
        print( 'Validation data ratio (external data): ', sdata._valid_perc,'%' )    
    
    
    # Create training and validation tables
    print( '\nTable splitting ....' )
    sdata._split()
    print( '.... done!' )
    
    print( '\nNumber of elements for training set: ', len( sdata._table_train[ sdata._col_paths ] ) )
    print( 'Number of IDs for training set: ', sdata._num_ids_train )
    
    if sdata._test_perc:    
        print( '\nNumber of elements for testing set: ', len( sdata._table_test[ sdata._col_paths ] ) )
        print( 'Number of IDs for testing set: ', sdata._num_ids_test )        
    
    if sdata._valid_perc:    
        print( '\nNumber of elements for validation set: ', len( sdata._table_valid[ sdata._col_paths ] ) )    
        print( 'Number of IDs for validation set: ', sdata._num_ids_valid )
        
        
    # Print balance of classes for each dataset
    print( '\nBalance training dataset (%): ', sdata._balance_train )
    if sdata._test_perc:
        print( 'Balance test dataset (%): ', sdata._balance_test )
    if sdata._valid_perc:
        print( 'Balance validation dataset (%): ', sdata._balance_valid )
        
            
    # Write tables to CSV files
    print( '\nWriting output tables ....' )
    sdata._write_tables( args.path_out , 
                         args.add_label )
    
    
    # Final print
    time2 = time.time()
    print( '\nElapsed time: ', time2-time1,'\n\n' )
    
    
    
    
# =============================================================================
# Call to main
# =============================================================================

if __name__ == '__main__':
    main()
    
    
