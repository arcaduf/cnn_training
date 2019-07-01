'''
Collect CSV into one CSV table
'''


# Author: Filippo Arcadu 
#         AIM4EYE Project
#         12/02/2018



from __future__ import division , print_function
import argparse
import sys , os
import glob
import pandas as pd
import random




# =============================================================================
# Parse input arguments
# =============================================================================

def _examples():
    print( '\n\nEXAMPLES\n\nCollect all CSV produced by slurming up a preprocessing function:\n' \
            '"""\npython cnn_collect_pred.py -i /pstore/data/pio/Tasks/PIO-233/data/RIDE_and_RISE_csme_qc_filt_mask_fov/\n"""\n\n'
           'Collect all CSV produced by slurming up a preprocessing function and randomly shuffle the order of the files:\n' \
            '"""\npython cnn_collect_pred.py -i /pstore/data/pio/Tasks/PIO-233/data/RIDE_and_RISE_csme_qc_filt_mask_fov/ -o /pstore/data/pio/Tasks/PIO-233/data/table_ride_and_rise_f2_qc_filt_fov.csv -s\n"""\n\n'
           'Do the same operation for multiple subfolders of a given folder:\n' \
            '"""\npython cnn_collect_pred.py -i /pstore/data/pio/Tasks/PIO-351/dl2/kfold_crossval/kfold_crossval_rtt_month12/predictions_ride_rise_valid/ -a\n"""\n\n'
          )    

          
          
def _get_args():
    parser = argparse.ArgumentParser(    
                                        prog='cnn_collect_pred',
                                        description='Collect all CSV produced by slurming up CNN predictions',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter
                                        ,add_help=False
                                    )
    
    parser.add_argument('-i', '--path_in', dest='path_in',
                        help='Specify input path with CSV prediction files')                        

    parser.add_argument('-s', '--shuffle', dest='shuffle', action='store_true',
                        help='Random shuffling of the rows') 

    parser.add_argument('-a', '--all', dest='all', action='store_true',
                        help='Do for all subfolders')                         

    parser.add_argument('-h', '--help', dest='help', action='store_true',
                        help='Print help and examples')                            

    args = parser.parse_args()
    
    if args.help is True:
        parser.print_help()
        _examples()
        sys.exit()

    if args.path_in is None:
        parser.print_help()
        sys.exit('\nERROR: Input path has not been specified!\n')  
  
    return args
    
    
    

# =============================================================================
# Get subfolders
# =============================================================================

def _get_subfolders( path_in ):
    return glob.glob( os.path.join( path_in , '*/' ) )

    
    
    
# =============================================================================
# Get list of CSV looking into the folder tree
# =============================================================================

def _get_list_csv( path_in ):
    list_csv = []
    
    for root, dirnames, filenames in os.walk( path_in ):
        for filein in filenames:
            if filein.endswith( '.csv' ) is True:
                filein = os.path.join( root , filein )
                list_csv.append( filein )

    return list_csv


    
    
# =============================================================================
# Write CSV collecting all individual CSVs
# =============================================================================

def _write_table( fileout , list_csv ):
    # Open output table file
    tb = open( fileout , 'w' )
    

    # For loop to read content from each CSV
    for i in range( len( list_csv ) ):
        fp = open( list_csv[i] , 'r' )
        
        if i == 0:
            header = fp.readline().strip( '\n' )
            tb.write( '%s' % header )
        else:
            fp.readline()
            
        content = fp.readline().strip( '\n' )
        tb.write( '\n%s' % content )
        fp.close()
    
    
    # Close table
    tb.close()
    print( '\nWritten ouput table: ', fileout )
    
    

    
# =============================================================================
# Main
# =============================================================================

def main():
    # Get input arguments
    args = _get_args()
    
    
    # Option "all" enabled
    if args.all:
        folders = _get_subfolders( args.path_in )
    else:
        folders = [ args.path_in ]
    
    
    # For loop 
    for i in range( len( folders ) ):
        # Construct output filename
        pathin = folders[i]
        if pathin[len(pathin)-1] == '/':
            pathin = pathin[:len(pathin)-1]
        fileout = pathin + '.csv'
   
        print( '\nInput path: ', pathin )
        print( 'Output CSV table: ', fileout )
    
        
        # Collect all CSV prediction files
        list_csv = _get_list_csv( pathin )
        print( 'Found ', len( list_csv ),' CSV files' )
    
        
        # Random shuffling if enabled
        if args.shuffle:
            random.shuffle( list_csv )
                                              
        
        # Write tables to CSV files
        _write_table( fileout , list_csv )
    
    print( '\n\n' )
    
    
    
    
# =============================================================================
# Call to main
# =============================================================================

if __name__ == '__main__':
    main()