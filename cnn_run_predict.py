'''
Wrapper to slurm-up the run of CNN predictions on a list of images
'''


# Author: Filippo Arcadu 
#         AIM4EYE Project
#         04/12/2017



from __future__ import print_function
import os
import sys
import argparse
import random
import pandas as pd
import numpy as np
import yaml
import glob



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
# Parse input arguments
# =============================================================================

def _examples():
    print( "\n\nEXAMPLE:\n" \
           "'''\npython cnn_run_predict.py -i cnn_run_predict.yml -l tmp/\n'''\n\n" )    
          


def _get_args():
    parser = argparse.ArgumentParser(
                                        prog='cnn_run_predict',
                                        description='Run DL forward predictions',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                        add_help=False
                                     )
    
    parser.add_argument('-i', '--input_yaml', dest='input_yaml',
                        help='Select input YAML file')

    parser.add_argument('-l', '--path_logs', dest='path_logs',
                        help='Select folder to place all logs')
                        
    parser.add_argument('-h', '--help', dest='help', action='store_true',
                        help='Print help and examples') 

    args = parser.parse_args()
    
    if args.help is True:
        parser.print_help()
        _examples()
        sys.exit()

    if args.input_yaml is None:
        parser.print_help()
        sys.exit('\nERROR ( _get_args ): Input YAML file has not been specified!\n')

    if os.path.isfile( args.input_yaml ) is None:
        parser.print_help()
        sys.exit('\nERROR ( _get_args ): Input YAML file does not exist!\n')        
        
    return args
    
    

# =============================================================================
# Class CNNRunPredict
# =============================================================================

class CNNRunPredict:
    
    # ===================================
    # Init 
    # ===================================

    def __init__( self , file_yaml ):
        # Read YAML
        self._read_yaml( file_yaml )
        
        
        # Get inputs
        self._get_inputs()
        
        
        # Get mode
        self._get_mode()
        
        
        # Get stem output folder
        self._get_output_folder()
        
        
        # Get DL models
        self._get_models()
        
        
        
    # ===================================
    # Read YAML file
    # ===================================

    def _read_yaml( self  , file_yaml):
        with open( file_yaml , 'r' ) as ymlfile:
            cfg = yaml.load( ymlfile )
        self._config = cfg
        
            
            
    # ===================================
    # Get inputs
    # ===================================

    def _get_inputs( self ):
        # Check YAML fields for the input data
        count = 0
        
        if 'folder' in self._config['input'].keys():
            field_folders = self._config['input']['folder']
        else:
            count += 1
            
        if 'csv' in self._config['input'].keys():
            field_csv = self._config['input']['csv']
        else:
            count += 1
            
        if count == 2:
            sys.exit( '\nERROR ( CNNRunPredict -- _get_inputs ): neither the subfield ' + \
                      '"folders" nor the subfield "csv" is present under the field "inputs"!\n')
                      
        if field_folders != 'None' and field_csv != 'None':
            sys.exit( '\nERROR ( CNNRunPredict -- _get_inputs ): both the subfield ' + \
                      '"folders" and the subfield "csv" are not None! Choose which one to use\n')
              
        elif field_folders != 'None' and field_csv == 'None':
            self._inputs  = field_folders
            self._folders = True
            
        elif field_folders == 'None' and field_csv != 'None':
            self._inputs  = field_csv
            self._folders = False
            
            if 'csv_col' not in self._config['input'].keys():
                sys.exit( '\nERROR ( CNNRunPredict -- _get_inputs ): the subfield ' + \
                          '"csv_col" has not been specified in the "input" field\n')
            else:
                self._col_file = self._config['input']['csv_col']
                
                
        # Check whether inputs is a list
        if isinstance( self._inputs , list ) is False:
            self._inputs = [ self._inputs ]
                
            
        # Check existence of the inputs
        for i in range( len( self._inputs ) ):
            if self._folders:
                if os.path.isdir( self._inputs[i] ) is False:
                    print( 'Warning folder ', self._inputs[i], 'does not exist!' )
            else:
                if os.path.isfile( self._inputs[i] ) is False:
                    print( 'Warning folder ', self._inputs[i], 'does not exist!' )


                    
    # ===================================
    # Get mode
    # ===================================                      
    
    def _get_mode( self ):
        # Get bottleneck features option
        if 'bottleneck' not in self._config['mode'].keys():
            self._bottleneck = False
        else:
            self._bottleneck = self._config['mode']['bottleneck']
            
            
        # Get architecture
        if 'arch' not in self._config['mode'].keys():
            self._arch = 'inception-v3'
        else:
            self._arch = self._config['mode']['arch']        



    # ===================================
    # Get output folder
    # ===================================                    
            
    def _get_output_folder( self ):
        if 'path' not in self._config['output'].keys():
            sys.exit( '\nERROR ( CNNRunPredict -- _get_output_folder ): the subfield ' + \
                      '"path" is not present under the field "output"!\n')
                      
        self._stem_out = self._config['output']['path']
        
        

    # ===================================
    # Get models
    # =================================== 
    
    def _get_models( self ):
        # Collect path/paths and hash/hashes
        if 'path' not in self._config['model'].keys():
            sys.exit( '\nERROR ( CNNRunPredict -- _get_models ): the subfield ' + \
                      '"path" is not present under the field "models"!\n')
        else:
            paths = self._config['model']['path']
            if isinstance( paths , list ):
                npaths = len( paths )
            else:
                npaths = 1
    
        if 'hash' not in self._config['model'].keys():
            sys.exit( '\nERROR ( CNNRunPredict -- _get_models ): the subfield ' + \
                      '"hash" is not present under the field "models"!\n')
        else:
            self._hashes = self._config['model']['hash']
            
            if isinstance( self._hashes , list ) is False:
                self._hashes = [ self._hashes ]
            
            if npaths == 1:
                paths = [ self._config['model']['path'] ] * len( self._hashes )
            
            
        # Collect models 
        self._models = []
        for i in range( len( paths ) ):
            search = glob.glob( os.path.join( paths[i] , '*' + str( self._hashes[i] ) + '*.h5' ) ) + \
                     glob.glob( os.path.join( paths[i] , '*' + str( self._hashes[i] ) + '*.hdf5' ) )
            file_hdf5 = search[0]    
      
            if len( file_hdf5 ):
                self._models.append( file_hdf5 )
            else:
                print( 'Warning: no HDF5 with hash ', self._hashes[i],' has been found in ', paths[i],'!' )
        
                      
    # ===================================
    # Run
    # =================================== 
    
    def _run( self , path_logs=None ):
        # Get path where to dump Slurm log files
        if path_logs is not None:
            if os.path.isdir( path_logs ) is False:
                os.mkdir( path_logs )
            os.chdir( path_logs )
        else:
            path_logs = './'
            
            
        # Base command
        program      = '/pstore/data/pio/ShellTools/parallel_processing.sh -y -a '
        comp         = '-s "--cpus-per-task=1 --ntasks=1 --qos short --time 00-02:00:00 --job-name cnn-predict"'
        modules      = '-m "python2,virtualenv"'
        env          = '-p "/pstore/data/pio/Tasks/PIO-40/myenv_latest_keras/"'
        base_command = program + ' ' + comp + ' ' + modules + ' ' + env        
        
        
        # Big for loop
        for i in range( len( self._inputs ) ):
            for j in range( len( self._models ) ):
                # Create tmp folder
                path_tmp = './tmp_pred'
                if os.path.isdir( path_tmp ):
                    path_tmp += '_' + str( random.getrandbits( 128 ) )
                path_tmp += '/'
                os.mkdir( path_tmp )

   
                # Get list of images
                if self._folders:
                    input   = self._inputs[i]
                    output  = path_tmp + 'list_imgs.csv'
                    command = '/pstore/data/pio/ShellTools/find_aim4eye_files.sh -p ' + input + ' -s "*" > ' + output
                    print( command )
                    os.system( command )
    
                else:
                    df        = pd.read_csv( self._inputs[i] , sep=SEP )
                    list_imgs = df[ self._col_file ].values 
                    df        = pd.DataFrame( { 'filepath': list_imgs } )
                    output    = path_tmp + 'list_imgs.csv'
                    df.to_csv( output , index=False )


                # Split CSV into batch-CSV
                input    = output
                output   = path_tmp + 'csv_parproc_imgs' 
                command  = '/pstore/data/pio/ShellTools/split_csv_file.sh ' + input + ' ' + output + ' 10'
                print( command )
                os.system( command )


                # Run parallel processing
                input  = output
                output = os.path.join( self._stem_out , 'preds_' + str( self._hashes[j] ) + '/' )
                p_curr = '/pstore/data/pio/Tasks/PIO-40/my-bitbucket/cnn-training-classification-and-regression/cnn-training/'

                if self._bottleneck:
                    command = base_command + ' "python ' + os.path.join( p_curr , 'cnn_predict.py' ) + ' -i INPUTPATH -o OUTPUTPATH -m ' + self._models[j] + ' -t ' + self._arch + ' -b " "' + input + '" "' + output + '"'
                else:
                    command = base_command + ' "python ' + os.path.join( p_curr , 'cnn_predict.py' ) + ' -i INPUTPATH -o OUTPUTPATH -m ' + self._models[j] + ' -t ' + self._arch + ' " "' + input + '" "' + output + '"'

                print( command )
                os.system( command )
        
        
        
                      
# =============================================================================
# Main
# =============================================================================

def main():
    # Get input arguments
    args = _get_args()
    
    
    # Read & analyze input YAML
    cnnr = CNNRunPredict( args.input_yaml )
    
    print( '\nNumber of inputs: ', len( cnnr._inputs ) )
    print( 'Inputs:\n' , cnnr._inputs )
    print( '\nNumber of models: ', len( cnnr._models ) )
    print( 'DL models:\n' , cnnr._models )
    print( '\nStem output folder:\n', cnnr._stem_out )
    print( '\nMode:' )
    print( '  Bottleneck features: ', cnnr._bottleneck )
    print( '  Architecture: ', cnnr._arch )
    
    
    # Run jobs
    print( '\n\nSending jobs to Slurm ....' )
    cnnr._run( path_logs=args.path_logs )
    
    print( '\n' )
    


# =============================================================================
# Call to main
# =============================================================================

if __name__ == '__main__':
    main()
