'''
Composite ROC curve for cross-fold validation
'''

from __future__ import print_function , division
import argparse
import sys , os
import json
import numpy as np
from sklearn import metrics as me
from scipy import interp
from metrics import Metrics




# =============================================================================
# MATPLOTLIB WITH DISABLED X-SERVER 
# =============================================================================

import matplotlib as mpl
mpl.use( 'Agg' )
import matplotlib.pyplot as plt




# =============================================================================
# MY VARIABLE FORMAT 
# =============================================================================

myint    = np.int
myfloat  = np.float32
myfloat2 = np.float64




# =============================================================================
# Get input arguments 
# =============================================================================

def _example():
    print( '\n\nPlot composite ROC curve from a CNN .json history file:\n' )
    print( '"""\npython cnn_roc_plot.py -i /pstore/data/pio/Tasks/PIO-457_GA-CNV-Fellow-Conv-MUV-from-CFPs/dl/grid_search/cnv/fellow/outputs/outputs_grid_point_016/\n"""\n\n' )


def _get_args():
    parser = argparse.ArgumentParser( description     = 'Plot ROC curve from CNN .json history file',
                                      formatter_class = argparse.ArgumentDefaultsHelpFormatter    ,
                                      add_help        = False )
    
    parser.add_argument('-i', '--path_in', dest='path_in',
                        help='Specify input path')
    
    parser.add_argument('-o', '--path_out', dest='path_out',
                        help='Specify output path')

    parser.add_argument('-h', '--help', dest='help', action='store_true',
                        help='Print example command line' )

    args = parser.parse_args()

    if args.help:
        parser.print_help()
        _example()
        sys.exit()

    if args.path_in is None:
        parser.print_help()
        sys.exit('\nERROR: Input path not specified!\n')

    return args




# =============================================================================
# Collect all JSON files
# =============================================================================

def collect_json_files( path ):
    # Case A: single JSON file provided in input
    if path.endswith( '.json' ):
        list_json = [ path ]


    # Case B: collect all JSON file from input path
    else:
        list_json = []
        for root, dirs, files in os.walk( path ):
            for name in files:
                if 'history' in name and '.json' in name:
                    list_json.append( os.path.join( root , name ) )

        list_json = sorted( list_json )

    return list_json




# =============================================================================
# Get data from JSON
# =============================================================================

def get_data_from_json( list_json ):
    trues = [];  probs = []
   
    for i in range( len( list_json ) ):
        df_json = json.loads( open( list_json[i] ).read() )

        trues.append( np.array( df_json[ 'y_true' ] ) )
        probs.append( np.array( df_json[ 'y_pred' ] ) )

    return trues , probs




# =============================================================================
# Create output path
# =============================================================================

def create_output_path( args ):
    if args.path_out is None:
        if args.path_in.endswith( '.json' ):
            path_out = os.path.dirname( args.path_in )
        else:
            path_out = args.path_in

    else:
        path_out = args.path_out

    return path_out




# =============================================================================
# Plot separate ROCs
# =============================================================================

def plot_separate_roc_curves( trues , probs , n_arr  , cmet , path_out ):
    # Initialize figure
    fig = plt.figure()
    string = str( n_arr ) + '-fold CV'
    plt.title( string + ' separate ROCs' , fontsize=16 ,fontweight='bold' )
    plt.xlabel( 'False Positive Rate' , fontsize=16 )
    plt.ylabel( 'True Positive Rate' , fontsize=16 )
    plt.xticks( fontsize=14 );  plt.yticks( fontsize=14 )


    # Plot "toss-coin" line y = x
    x_line = np.linspace( 0 , 1 , 50 )
    y_line = x_line.copy()
    plt.plot( x_line , y_line , '--' , lw=2 , color='black' )
    

    # Main for loop
    for i in range( n_arr ):
        # Get arrays
        y_score = np.array( probs[i] )
        y_true  = np.array( trues[i] )

        
        # Compute all metrics
        cmet._run( y_true , y_score )


        # Compute ROC AUC score
        roc_auc = me.auc( cmet._fpr , cmet._tpr )
    
        
        # Get confidence interval
        cmet._ci_bstrap( y_true , y_score )


        # Get Youden's point for fpr and tpr
        if len( cmet._fpr_best ) > 1:
            cmet._fpr_best = fpr_best[0]

        if len( cmet._tpr_best ) > 1:
            cmet._tpr_best = tpr_best[0]


        # Create label
        label = 'Fold n.' + str( i ) + \
                ': AUC = ' + str( round( roc_auc , 2 ) ) + \
                ' , 95%CI= [' + str( round( cmet._auc_ci_down , 2 ) ) + ',' + \
                str( round( cmet._auc_ci_up , 2 ) ) + ']'
        
        
        # Plot ROC curve
        plt.plot( cmet._fpr , cmet._tpr , lw=2 , label=label )

              
    # Save figure
    plt.tight_layout()
    plt.legend( fontsize=10 )
        
    save_plot = os.path.join( path_out , 'separate_roc_curve.png' )
        
    if '.svg' in save_plot:
        plt.savefig( save_plot , format='svg' , dpi=1200 )
    else:
        plt.savefig( save_plot )

    print( '\nSeparate ROC plots saved to:\n', save_plot )
        
        
        

# =============================================================================
# Plot composite ROC
# =============================================================================

def plot_composite_roc_curve( trues , probs , n_arr  , cmet , path_out ):
    # Collect all TPR and FPR
    tpr_list  = []
    auc_list  = []
    sens_list = []
    spec_list = []
    
    max_length = 0

    mean_fpr = np.linspace(0, 1, 100)
    
    for i in range( n_arr ):
        # Get individual y_score and y_true
        y_score = np.array( probs[i] )
        y_true  = np.array( trues[i] )
                    
    
        # Compute metrics
        cmet._run( y_true , y_score )

        tpr_list.append( interp( mean_fpr , cmet._fpr , cmet._tpr ) )
        tpr_list[-1][0] = 0.0
        auc_list.append( cmet._auc )
        sens_list.append( cmet._sensitivity )
        spec_list.append( cmet._specificity )
        
        if len( cmet._fpr ) > max_length:
            max_length = len( cmet._fpr )
            x_com      = cmet._fpr.copy()
        
        
    # Convert array to list    
    tpr_list  = np.array( tpr_list )    
    auc_list  = np.array( auc_list ) 
    sens_list = np.array( sens_list )    
    spec_list = np.array( spec_list )     
        
        
    # Construct the 3 curves
    mean_tpr     = np.mean( tpr_list , axis=0)
    mean_tpr[-1] = 1.0
    mean_auc     = me.auc( mean_fpr , mean_tpr )
    std_auc      = np.std( auc_list )

    std_tpr    = np.std( tpr_list , axis=0 )
    tprs_upper = np.minimum( mean_tpr + std_tpr , 1 )
    tprs_lower = np.maximum( mean_tpr - std_tpr , 0 )
 

    # Initialize figure
    fig = plt.figure()
    string = str( n_arr ) + '-fold CV'
    plt.title( string + ' composite ROC' , fontsize=16 ,fontweight='bold' )
    plt.xlabel( 'False Positive Rate' , fontsize=16 )
    plt.ylabel( 'True Positive Rate' , fontsize=16 )
    plt.xticks( fontsize=14 );  plt.yticks( fontsize=14 )


    # Plot "toss-coin" line y = x
    x_line = np.linspace( 0 , 1 , 50 )
    y_line = x_line.copy()
    plt.plot( x_line , y_line , '--' , lw=2 , color='red' )
        
        
    # Create additional text
    auc_mean  = np.mean( auc_list );  auc_std = np.std( auc_list )
    sens_mean = np.mean( sens_list );  sens_std = np.std( sens_list )
    spec_mean = np.mean( spec_list );  spec_std = np.std( spec_list )
    
    string = 'AUC  = ' + str( round( auc_mean , 2 ) ) + ' +/- ' + str( round( auc_std , 2 ) ) + '\n' + \
             'SENS = ' + str( round( sens_mean , 2 ) ) + ' +/- ' + str( round( sens_std , 2 ) ) + '\n' + \
             'SPEC = ' + str( round( spec_mean , 2 ) ) + ' +/- ' + str( round( spec_std , 2 ) )            
    
    plt.text( 0.60 , 
              0.20 , 
              string , 
              fontsize=13 , 
              horizontalalignment='left' ,
              verticalalignment='top' ,
              color='black' )       
        
        
    # Plot ROC curve
    plt.plot( mean_fpr , mean_tpr , lw=2 , color='black' , label='Mean ROC curve' )

    
    # Color area between min and max ROC
    plt.fill_between( mean_fpr , tprs_lower , tprs_upper , color='grey' , 
                      alpha=.25 , label=r'$\pm$ 1 std. dev.' )
          
    
    # Save figure
    plt.tight_layout()
    plt.legend( fontsize=10 )
    
    save_plot = os.path.join( path_out , 'composite_roc_curve.png' )
        
    if '.svg' in save_plot:
        plt.savefig( save_plot , format='svg' , dpi=1200 )
    else:
        plt.savefig( save_plot )

    print( '\nComposite ROC curve saved to:\n' )
    print( save_plot )
        
        
     

# =============================================================================
# Plot single ROC
# =============================================================================

def plot_single_roc_curve( trues , probs  , cmet , output_stem ):
    # Initialize figure
    fig = plt.figure()
    plt.title( 'ROC curve' , fontsize=16 ,fontweight='bold' )
    plt.xlabel( 'False Positive Rate' , fontsize=16 )
    plt.ylabel( 'True Positive Rate' , fontsize=16 )
    plt.xticks( fontsize=14 );  plt.yticks( fontsize=14 )


    # Plot "toss-coin" line y = x
    x_line = np.linspace( 0 , 1 , 50 )
    y_line = x_line.copy()
    plt.plot( x_line , y_line , '--' , lw=2 , color='black' )
    

    # Get arrays
    y_score = np.array( probs[0] )
    y_true  = np.array( trues[0] )
        
        

    # Get balance
    perc_class0 = len( y_true[ y_true == 0 ] ) * 1.0 / len( y_true ) * 100.0
    perc_class1 = 100 - perc_class0


    # Compute all metrics
    cmet._run( y_true , y_score )


    # Compute ROC AUC score
    roc_auc = me.auc( cmet._fpr , cmet._tpr )
    
    
    # Get confidence interval
    cmet._ci_bstrap( y_true , y_score )
    
    
    # Get Youden's point for fpr and tpr
    if len( cmet._fpr_best ) > 1:
        cmet._fpr_best = fpr_best[0]

    if len( cmet._tpr_best ) > 1:
        cmet._tpr_best = tpr_best[0]


    # Create label
    label = 'AUC = ' + str( np.round( roc_auc , 3 ) ) + \
            ' , 95% CI = [' + str( np.round( cmet._auc_ci_down , 3 ) ) + ',' + \
            str( np.round( cmet._auc_ci_up , 3 ) ) + ']\n' + \
            'SENS( Youden ) = ' + str( np.round( cmet._sensitivity , 3 ) ) + \
            ' , 95% CI = [' + str( np.round( cmet._sensitivity_ci_down , 3 ) ) + ',' + \
                        str( np.round( cmet._sensitivity_ci_up , 3 ) ) + ']\n' + \
            'SPEC( Youden ) = ' + str( np.round( cmet._specificity , 3 ) ) + \
                        ' , 95% CI = [' + str( np.round( cmet._specificity_ci_down , 3 ) ) + ',' + \
                        str( np.round( cmet._specificity_ci_up , 3 ) ) + ']\n' + \
            'THRES( Youden ) = ' + str( cmet._thres_best[0] ) + '\n' + \
            'Balance = ' + str( np.round( perc_class1 , 2 ) ) + '% ( Class 1 )'

        
        
    # Plot ROC curve
    plt.plot( cmet._fpr , cmet._tpr , lw=2 )


    # Add operating point
    plt.plot( cmet._fpr_best , cmet._tpr_best , marker='o', 
              markersize=8 , color='red' )
    plt.text( cmet._fpr_best -0.02 , cmet._tpr_best + 0.05 , 'OP' , fontweight='bold' )

    # Color area between min and max ROC
    #plt.fill_between( cmet._fpr , cmet._tpr_ci_down , cmet._tpr_ci_up , 
    #                  color='grey' , alpha=.25 , label=r'95% CI' )
 

    # Add text to plot
    plt.text( 0.31 , -0.01 , label )


    # Print to standard output
    print( '\nAUC = ',  roc_auc , ' , 95%CI= [' , cmet._auc_ci_down , ',' , cmet._auc_ci_up , ']' )
    print( 'SENS( Youden ) = ',  cmet._sensitivity , ' , 95%CI= [' , cmet._sensitivity_ci_down , ',' , cmet._sensitivity_ci_up , ']' )
    print( 'SPEC( Youden ) = ',  cmet._specificity , ' , 95%CI= [' , cmet._specificity_ci_down , ',' , cmet._specificity_ci_up , ']' )
    print( 'THRES( Youden ) = ', cmet._thres_best )
    print( 'Balance = ', perc_class0,'% ( Class 0 )  --  ', perc_class1,'% ( Class 1 )'  )

    
    # Save figure
    plt.tight_layout()
    plt.legend( fontsize=10 )
    
    save_plot = os.path.join( output_stem , 'single_roc_curve.png' )
        
    if '.svg' in save_plot:
        plt.savefig( save_plot , format='svg' , dpi=1200 )
    else:
        plt.savefig( save_plot )

    print( '\nSingle ROC plot saved to:\n', save_plot )
 



# =============================================================================
# MAIN
# =============================================================================

def main():
    # Get input arguments
    args = _get_args()
    print( '\n----------------------------------------------' )
    print( '    Create ROC plot for classification model    ' )
    print( '----------------------------------------------\n' )

    
    # Collect all JSON files
    list_json = collect_json_files( args.path_in )
    n_arr     = len( list_json )
    print( '\nNumber of folds: ', n_arr )
    print( '\nInput JSON history files: ', list_json )


    # Get list of ground truth and probabilities
    trues , probs = get_data_from_json( list_json )


    # Initialize metrics class
    cmet = Metrics( task      = 'class' ,
                    n_classes = 2       ) 


    # Create output stem
    path_out = create_output_path( args )


    # Plot ROC curves
    if n_arr > 1:
        print( '\nPlot separate ROC curve ....' )
        plot_separate_roc_curves( trues , probs , n_arr , cmet , path_out )

        print( '\nPlot composite ROC curve ....' )
        plot_composite_roc_curve( trues , probs , n_arr , cmet , path_out )

    else:
        print( '\nPlot single ROC curve ....' )
        plot_single_roc_curve( trues , probs , cmet , path_out )
    
    print( '\n\n' )




# =============================================================================
# CALL TO MAIN
# =============================================================================

if __name__ == '__main__':
    main()

