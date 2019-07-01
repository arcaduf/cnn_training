'''
Evaluate grid search with cross-fold validation
'''

# Author: Filippo Arcadu
# Date: 21.01.2019
# PHCOSS -- PIO-442


from __future__ import division , print_function
import argparse
import sys , os , glob , json
import numpy as np
import pandas as pd




# =============================================================================
# CSV separator
# =============================================================================

SEP= ','




# =============================================================================
# Parse input arguments
# =============================================================================

def _examples():
    print( "\n\nEXAMPLE\n\nEvaluate 5-fold CV grid search:\n\n" \
            "'''\npython evaluate_grid_search.py -i /pstore/data/pio/Tasks/PIO-457_GA-CNV-Fellow-Conv-MUV-from-CFPs/dl/grid_search/cnv/fellow/outputs/\n'''\n\n" )    
          


def _get_args():
    parser = argparse.ArgumentParser(
                                        prog='evaluate_grid_search',
                                        description='Evaluate cross-fold valiadated grid-search',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                        add_help=False
                                     )
    
    parser.add_argument('-i', '--path_in', dest='path_in',
                        help='Input path')                                            
    
    parser.add_argument('-o', '--file_out', dest='file_out',
                        help='Output CSV')                                            
    
    parser.add_argument('-h', '--help', dest='help', action='store_true',
                        help='Print help and examples')                            

    args = parser.parse_args()
    
    if args.help is True:
        parser.print_help()
        _examples()
        sys.exit()

    if args.path_in is None:
        parser.print_help()
        sys.exit('\nERROR ( _get_args ): Input path has not been specified!\n')
        
    if os.path.isdir( args.path_in ) is False:
        parser.print_help()
        sys.exit('\nERROR ( _get_args ): Input path ' + args.path_in + ' does not exist!\n')
        
    return args




# =============================================================================
# Get AUC from JSON files
# =============================================================================

def get_auc_from_json( file_json ):
    df = json.loads( open( file_json ).read() )

    auc_tl = df[ 'peak_auc_valid' ]
    auc_ft = df[ 'peak_auc_valid_ft' ]

    if auc_tl < auc_ft:
        auc = auc_ft
    else:
        auc = auc_tl

    return auc




# =============================================================================
# Get AUC from CSV files
# =============================================================================

def get_auc_from_csv( file_csv ):
    df = pd.read_csv( file_csv , sep=SEP )

    auc_arr = df[ 'val_auc' ].values
    auc     = np.max( auc_arr )

    return auc




# =============================================================================
# Analyzing grid search
# =============================================================================

def analyze_grid_search( path ):
    # Check path
    if 'outputs' not in path:
        aux  = os.path.join( path , 'outputs/' )
        path = aux

    
    # Get all grid search folds
    list_gsearch = sorted( glob.glob( os.path.join( path , 'outputs_grid_point_*' ) ) )
    print( '\nNumber of retrieved grid points: ', len( list_gsearch ) )


    # Initialize data frame
    df = pd.DataFrame( columns=[ 'Grid search point' , 'Mean AUC' , 'Std AUC' , 
                                 'AUC fold 00' , 'AUC fold 01' , 'AUC fold 02' ,
                                 'AUC fold 03' , 'AUC fold 04'] )


    # For loop on the grid point folders
    for i in range( len( list_gsearch ) ):
        # Initialize auxiliary list
        auc = []
        
        
        # Collect all AUC values
        n_folders = 0
        for _ , dirnames , filenames in os.walk( list_gsearch[i] ):
            n_folders += len( dirnames )
            
        flag_complete = 1

        for j in range( n_folders ):
            # Get path of the fold
            if j < 10:
                str_num = '0' + str( j )
            else:
                str_num = str( j )
            path_kfold = os.path.join( list_gsearch[i] , 'outputs_kf0' + str_num )

            # Look for JSON files
            file_json = glob.glob( os.path.join( path_kfold , '*history*.json' ) )

            if len( file_json ):
                file_json = file_json[0]
                flag      = 1
            else:
                file_csv      = glob.glob( os.path.join( path_kfold , '*logger*' ) )[0]
                flag          = 0
                flag_complete = 0

            # Get AUC value
            if flag:
                value = get_auc_from_json( file_json )
            else:
                value = get_auc_from_csv( file_csv )
            
            auc.append( value )

        print( '\n\tFound ', len( auc ), ' AUC values inside ', list_gsearch[i] )

        
        # Compute mean and std AUC
        auc    = np.array( auc )
        v_mean = np.mean( auc )
        v_std  = np.std( auc )

        if flag_complete:
            print( '\tAUC = ', v_mean,' +/- ', v_std )
        else:
            print( '\tAUC = ', v_mean,' +/- ', v_std , '  -->  Warning: some jobs not complete!' )

        df.loc[ i ] = [ i , v_mean , v_std ] + auc.tolist()

    return df




# =============================================================================
# Identify best point
# =============================================================================

def identify_best_point( df ):
    auc      = df[ 'Mean AUC' ].values
    auc_best = np.max( auc )
    ind_best = np.argwhere( auc == auc_best )[0]

    print( '\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n' )
    print( 'Best grid point: n.', df[ 'Grid search point' ].values[ind_best] )
    print( 'AUC = ', df[ 'Mean AUC' ].values[ind_best] , ' +/- ' , df[ 'Std AUC' ].values[ind_best] )
    print( '\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n' )




# =============================================================================
# Save output
# =============================================================================

def save_output( args , df ):
    # Output path
    if args.file_out is None:
        path_out = os.path.abspath( args.path_in )
        file_out = os.path.join( path_out , 'results_grid_search.csv' )
    
    else:
        if os.path.isdir( args.path_out ) is False:
            path_out = args.path_in
            file_out = os.path.join( path_out , 'results_grid_search.csv' )
        else:
            file_out = args.file_out


    # Save to CSV
    df.to_csv( file_out , sep=SEP , index=False )
    print( '\nResults saved to:\n', file_out )




# =============================================================================
# Main
# =============================================================================

def main():
    # Get input argument
    args = _get_args()


    # Prints
    print( '\n-------------------------------------------------------' )
    print( '       Evaluate cross-fold validated grid search         ' )
    print( '-------------------------------------------------------\n' )

    print( '\nInput path:\n', args.path_in )


    # Analyze grid search
    print( '\nAnalyzing grid search ....' )
    df = analyze_grid_search( args.path_in )


    # Identify grid point with best mean AUC
    identify_best_point( df )


    # Save data frame to CSV
    save_output( args , df )

    print( '\n\n' )




# =============================================================================
# Call to Main
# =============================================================================

if __name__ == '__main__':
    main()
