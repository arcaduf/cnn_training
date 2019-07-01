'''
Collate ground truth and predictions into a JSON file
'''

# Author: Filippo Arcadu
# Date: 14.02.2019


from __future__ import division , print_function
import pandas as pd
import os , json



# Inputs
csv_preds = '/pstore/data/pio/Tasks/PIO-471_DL-to-identify-laterality-CFPs/dl/preds/preds_central_field.csv'
csv_trues = '/pstore/data/pio/Tasks/PIO-471_DL-to-identify-laterality-CFPs/dl/split_tables/rr_am_valid.csv'
col_imgs  = 'Filepath Crop'
col_label = 'Laterality-Binary'
SEP       = ','



# Read and merge data frames
df_preds = pd.read_csv( csv_preds , sep=SEP )
print( '\nPrediction data frame shape: ', df_preds.shape )

df_trues = pd.read_csv( csv_trues , sep=SEP )[ [ col_imgs , col_label ] ]
print( '\nGround Truth data frame shape: ', df_trues.shape )

df_trues = df_trues.rename( columns={ col_imgs: 'image' } )

df_merge = pd.merge( df_preds , df_trues , on=[ 'image' ] )
print( '\nMerged data frame: ', df_merge.shape )



# Create JSON file
y_trues = df_merge[ col_label ].values.tolist()
y_preds = df_merge[ [ 'class n.0 probability' , 'class n.1 probability' ] ].values.tolist()

df_json = { 'y_true': y_trues ,
            'y_pred': y_preds }

print( '\ny_trues:\n' , y_trues )
print( '\ny_preds:\n' , y_preds )

file_out = os.path.splitext( csv_preds )[0] + '.json'

with open( file_out , 'w' ) as fp:
    json.dump( df_json , fp )

print( '\nWritten output JSON: ', file_out )
print( '\n\n' )
