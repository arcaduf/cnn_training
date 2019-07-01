'''
Data augmentation for CNN training
'''


# Author: Filippo Arcadu 
#         AIM4EYE Project
#         19/01/2018




from __future__ import division , print_function
import argparse
import sys
import os
import numpy as np
import pandas as pd
import yaml
import time
import random
import glob

from skimage import io

from imgaug import augmenters as iaa

import multiprocessing as mproc




# =============================================================================
# Names of available transformations for augmentation 
# =============================================================================

TRANSFORM_KEYS = [ 'crop' , 'flip_lr' , 'flip_ud' ,'scaling' , 'translation' , 
                   'rotation' , 'shear' , 'gaussian_blur' , 'average_blur' , 
                   'median_blur' , 'sharpen' , 'emboss' , 'dropout' ,
                   'brightness' , 'hue_and_saturation' ,
                   'contrast_normalization' , 'elastic_transformation' ] 




# =============================================================================
# Types 
# =============================================================================

myfloat = np.float32
myint   = np.int




# =============================================================================
# Just a "big" number 
# =============================================================================

NUM_BIG = 1e10




# =============================================================================
# Parse input arguments
# =============================================================================

def _examples():
    print( '\n\nEXAMPLES\n\nData augmentation for transfer-learning:\n' \
           'python cnn_data_augment.py -i /pstore/data/pio/Tasks/PIO-230/dl/dataset_2classes/train_augm/ -p /pstore/data/pio/Tasks/PIO-230/dl/configs/cnn_data_augment_2classes.yml -f png\n\n'
          ) 

          
          
def _get_args():
    parser = argparse.ArgumentParser(   
                                        prog='cnn_data_augment',
                                        description='Data augmentation for transfer-learning',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter
                                        ,add_help=False
                                    )
    
    parser.add_argument('-i', '--path_in', dest='path_in',
                        help='Specify input training path')

    parser.add_argument('-p', '--config_file', dest='config_file',
                        help='Specify input YAML config file')
                        
    parser.add_argument('-f', '--img_format', dest='img_format',
                        help='Specify input image format')

    parser.add_argument('-n', '--num_cores', dest='num_cores', type=myint, default=20 ,
                        help='Specify number of cores to use for multiprocessing')

    parser.add_argument('--inplace', dest='inplace', action='store_true',
                        help='Augment dataset directly on the training folder')

    parser.add_argument('-h', '--help', dest='help', action='store_true',
                        help='Print help and examples')                        

    args = parser.parse_args()
    
    if args.help is True:
        parser.print_help()
        _examples()
        sys.exit()

    if args.path_in is None:
        parser.print_help()
        sys.exit('\nERROR: Input training path not specified!\n')

    if os.path.isdir( args.path_in ) is False:
        parser.print_help()
        sys.exit('\nERROR: Input training path does not exist!\n')        

    if args.config_file is None:
        parser.print_help()
        sys.exit('\nERROR: Path to YAML config file not specified!\n')    
        
    return args
    
    
    

# =============================================================================
# Augment images, function used by the slaves of multiprocessing
# =============================================================================

def augment_image_mproc( ind , num_imgs , list_imgs ,
                         seq , ext_img , ext_out ):
    # Pick a random index
    rand_int = np.random.choice( num_imgs )

 
    # Load random image
    filein      = list_imgs[rand_int]
    pathout , _ = os.path.split( os.path.abspath( filein ) )
    basename    = os.path.basename( filein )
    img         = load_img( filein , ext_img )


    # Perform the augmentation
    seq.reseed( np.random.choice( NUM_BIG ) )
    img_augm  = seq.augment_image( img )


    # Save augmented image and mask
    save_image( img_augm , pathout , basename , ext_out )


       
       
# =============================================================================
# Load image
# =============================================================================
   
def load_img( file_img , ext ):
    # Case A: .npy file
    if ext == '.npy':
        img = np.load( file_img )


    # Case B: any other format
    else:
        img = io.imread( file_img )   

    return img




# =============================================================================
# Save augmented image
# =============================================================================

def save_image( img , pathout , basename , ext_out ):
    # Create output filenames
    fileout = create_output_filename( pathout , basename , ext_out )
    
    
    # Case A: output extension is .npy 
    if ext_out == '.npy':
        np.save( fileout , img )


    # Case B: output extension is anything else
    else:
        io.imsave( fileout , img )

        
        
        
# =============================================================================
# Create output filename
# =============================================================================

def create_output_filename( pathout , basename , ext_out ):
    hash = str( random.getrandbits( 128 ) )

    fileout  = os.path.join( pathout , basename + '_' + hash + ext_out )

    return fileout

    
    
    
# =============================================================================
# Definition of class for data augmentation
# =============================================================================

class DataAugm:

    # ===================================
    # Init 
    # ===================================
    
    def __init__( self , 
                  path_in , 
                  config_file ,
                  num_cores = 20 ,
                  ext       = '.png' ,
                  inplace   = False ):
                  
        # Assign inputs to class attributes
        self._path_in     = path_in
        self._config_file = config_file
        self._num_cores   = num_cores
        self._ext_in      = ext


        # Inplace option
        if inplace is False:
            self._create_copy_directory()

       
        # Correct input image format if needed
        if self._ext_in[0] != '.':
            self._ext_in = '.' + self._ext_in
        

        # Read input table
        self._get_list_images_and_labels()
        
        
        # Load yaml config file
        self._load_config()

        
        # Check output format 
        self._check_and_correct_format()
        
    

    # ===================================
    # Create copy of directory
    # ===================================

    def _create_copy_directory( self ):
        pathin = self._path_in
        
        if pathin[len(pathin)-1] == '/':
            pathin = pathin[:len(pathin)-1]
            
        pathin2 = pathin + '_augm/'
        
        command = 'cp -r ' + pathin + ' ' + pathin2
        os.system( command )
        
        self._path_in = pathin2

    
    
    # ===================================
    # Get list of images and labels
    # ===================================

    def _get_list_images_and_labels( self ):
        # Get first level folder inside input path
        for root, dirnames, filenames in os.walk( self._path_in ):
            break
        dirnames         = sorted( dirnames )
        
        self._path_class = []
        for i in range( len( dirnames ) ):
            self._path_class.append( self._path_in + dirnames[i] + '/' )
        
        self._num_classes = len( self._path_class )        


        # Get list of images
        self._list_imgs_class   = []
        self._num_imgs_class    = []
        self._list_imgs         = []
        self._num_imgs          = []
        
        for i in range( self._num_classes ):
            list_imgs_class  = sorted( glob.glob( self._path_class[i] + '*' + self._ext_in ) )
            self._list_imgs_class.append( list_imgs_class )
            self._num_imgs_class.append( len( list_imgs_class ) )
            self._list_imgs += list_imgs_class

        self._num_imgs = len( self._list_imgs )
    

        # Get common path among files
        self._pathout_table = os.path.commonprefix( self._list_imgs )


            
    # ===================================
    # Check and correct output format
    # ===================================
    
    def _check_and_correct_format( self ):
        self._ext_out = self._ext_out.lower()
    
        if self._ext_out[0] != '.':
            self._ext_out = '.' + self._ext_out
            
        if self._ext_out not in [ '.npy' , '.tif' , '.tiff' ,
                                        '.jpg' , '.jpeg' , '.png' ]:
            sys.exit( '\nERROR: output format ' + self._ext_out + ' not available!\n' + \
                      'Choose among: .npy , .tif , .tiff , .jpg , .jpeg , .png' )
    
    
    
    # ===================================
    # Load YAML config file
    # ===================================
    
    def _load_config( self ):
       # Initialize empty list of transformations
       self._transform_names  = []
       self._transform_params = []
        
        
       # Load config file if not None
       with open( self._config_file , 'r' ) as ymlfile:
           cfg = yaml.load( ymlfile )
   
   
       # General setting
       self._num_samples       = cfg['general']['num_samples']
       self._ext_out           = cfg['general']['ext_out'] 
       self._class_weights     = cfg['general']['class_weights']       
        

       # Transform class weights into floats
       chunks              = np.array( self._class_weights.split( ':' ) ).astype( myfloat )
       self._class_weights = chunks 


       # Get number of samples per class to create
       weight_tot              = self._class_weights.sum()
       self._num_samples_class = []
       
       for i in range( self._num_classes ):
           self._num_samples_class.append( myint( 1.0 * self._num_samples * self._class_weights[i]  / weight_tot ) )
   
                
       # Assign attributes to class fields
       for key in TRANSFORM_KEYS:        
           if cfg.has_key( key ) and cfg[key]['enable']:
               self._transform_names.append( key )
               
               if key == 'flip_lr' or key == 'flip_ud':
                   dict = { 'prob' : cfg[key]['prob'] }
   
               else:
                   dict = { 'prob' : cfg[key]['prob'] ,
                            'range': cfg[key]['range'] }

               if key == 'sharpen' or key == 'emboss' or key == 'elastic_transformation':
                  dict.update( { 'alpha': cfg[key]['alpha'] } )

               self._transform_params.append( dict )



    # ===================================
    # Do data augmentation
    # ===================================

    def _do_augment( self ):
       
       # Create imgaug sequence
       self._create_imgaug_sequence()
   
         
       # For loop
       print( '\nAugmentating dataset ....' )

       for i in range( self._num_classes ):
           if self._num_cores > 1:    
               pool = mproc.Pool( processes=self._num_cores )
               [ pool.apply_async( augment_image_mproc , 
                                   args=( j , self._num_imgs_class[i] , self._list_imgs_class[i] , 
                                          self._seq_imgaug , self._ext_in , self._ext_out ) ) \
                 for j in range( self._num_samples_class[i] ) ]
               pool.close()
               pool.join()
   
           else:
               for j in range( self._num_samples_class[i] ):
                   self._augment_image( i , j )         



    # ===================================
    # Create IMGAUG sequence
    # ===================================
    
    def _create_imgaug_sequence( self ):
       # Initialize empty list of transformations
       seq_imgaug = []
   
  
       # For loop to collect all transformations   
       for i in range( len( self._transform_names ) ):
          # Get probability
          prob = self._cast_parameter( self._transform_params[i][ 'prob' ] , 
                                       format='float' )
                   
          # Horizontal flipping
          if self._transform_names[i] == 'flip_lr':
              seq_imgaug.append( iaa.Fliplr( prob ) )
                
          
          # Vertical flipping
          elif self._transform_names[i] == 'flip_ud':
              seq_imgaug.append( iaa.Flipud( prob ) )


          # Cropping and padding
          elif self._transform_names[i] == 'crop':
             r = self._cast_parameter( self._transform_params[i][ 'range' ] , 
                                       format='tuple' )
             seq_imgaug.append( iaa.Sometimes( prob ,
                                                    iaa.CropAndPad( percent=r ) 
                                                   )
                              )
            
          # Scaling
          elif self._transform_names[i] == 'scaling':
             r = self._cast_parameter( self._transform_params[i][ 'range' ] , 
                                       format='tuple_double' )
             seq_imgaug.append( iaa.Sometimes( prob , 
                                                    iaa.Affine( scale=r ,
                                                                order=[1] ) 
                                             )
                              )
            
          # Translation
          elif self._transform_names[i] == 'translation':
             r = self._cast_parameter( self._transform_params[i][ 'range' ] , 
                                       format='tuple_double' )
             seq_imgaug.append( iaa.Sometimes( prob , 
                                                    iaa.Affine( translate_percent=r ,
                                                                order=[1] )                                                               
                                             )
                              )

          # Rotation
          elif self._transform_names[i] == 'rotation':
             r = self._cast_parameter( self._transform_params[i][ 'range' ] , 
                                       format='tuple' )
             seq_imgaug.append( iaa.Sometimes( prob , 
                                                    iaa.Affine( rotate=r ,
                                                                order=[1] ) 
                                             )
                              )

          # Shearing
          elif self._transform_names[i] == 'shearing':
             r = self._cast_parameter( self._transform_params[i][ 'range' ] , 
                                       format='tuple' )
             seq_imgaug.append( iaa.Sometimes( prob , 
                                               iaa.Affine( shear=r ,
                                                           order=[1] ) 
                                             )
                              )

          # Elastic deformation
          elif self._transform_names[i] == 'elastic_transformation':
             a = self._cast_parameter( self._transform_params[i][ 'alpha' ] , 
                                       format='tuple' )
             s = self._cast_parameter( self._transform_params[i][ 'range' ] , 
                                       format='float' )
             seq_imgaug.append( iaa.Sometimes( prob ,
                                               iaa.ElasticTransformation( alpha=a ,
                                                                          sigma=s )
                                             )
                              )

          # Gaussian blurring
          elif self._transform_names[i] == 'gaussian_blur':
             r = self._cast_parameter( self._transform_params[i][ 'range' ] , 
                                       format='tuple' )
             seq_imgaug.append( iaa.Sometimes( prob , 
                                               iaa.GaussianBlur( r ) 
                                             )
                              )
            
          # Average blurring
          elif self._transform_names[i] == 'average_blur':
             r = self._cast_parameter( self._transform_params[i][ 'range' ] , 
                                       format='tuple' )
             seq_imgaug.append( iaa.Sometimes( prob ,
                                               iaa.AverageBlur( k=r )
                                             )
                              )

          # Median blurring
          elif self._transform_names[i] == 'median_blur':
             r = self._cast_parameter( self._transform_params[i][ 'range' ] , 
                                       format='tuple' )
             seq_imgaug.append( iaa.Sometimes( prob , 
                                               iaa.MedianBlur( k=r )
                                             )
                              )

          # Sharpening
          elif self._transform_names[i] == 'sharpen':
             a = self._cast_parameter( self._transform_params[i][ 'alpha' ] , 
                                       format='tuple' )             
             l = self._cast_parameter( self._transform_params[i][ 'range' ] , 
                                       format='tuple' )
             seq_imgaug.append( iaa.Sometimes( prob , 
                                               iaa.Sharpen( alpha=a ,
                                                            lightness=l ) 
                                             )
                              ) 

          # Emboss
          elif self._transform_names[i] == 'emboss':
             a = self._cast_parameter( self._transform_params[i][ 'alpha' ] , 
                                       format='tuple' )             
             s = self._cast_parameter( self._transform_params[i][ 'range' ] , 
                                       format='tuple' )
             seq_imgaug.append( iaa.Sometimes( prob , 
                                               iaa.Emboss( alpha=a ,
                                                           strength=s )
                                             )
                              )
            
          # Dropout
          elif self._transform_names[i] == 'dropout':
             r = self._cast_parameter( self._transform_params[i][ 'range' ] , 
                                       format='tuple' )
             seq_imgaug.append( iaa.Sometimes( prob , 
                                               iaa.Dropout( r ) 
                                             )
                              )

          # Brightness
          elif self._transform_names[i] == 'brightness':
             r = self._cast_parameter( self._transform_params[i][ 'range' ] , 
                                       format='tuple' )
             seq_imgaug.append( iaa.Sometimes( prob , 
                                               iaa.Add( r )
                                             )
                              )

          # Hue and saturation
          elif self._transform_names[i] == 'hue_and_saturation':
             r = self._cast_parameter( self._transform_params[i][ 'range' ] , 
                                       format='tuple' )
             seq_imgaug.append( iaa.Sometimes( prob , 
                                               iaa.AddToHueAndSaturation( r )
                                             )
                              )

          # Contrast normalization
          elif self._transform_names[i] == 'contrast_normalization':
             r = self._cast_parameter( self._transform_params[i][ 'range' ] , 
                                       format='tuple' )
             seq_imgaug.append( iaa.Sometimes( prob , 
                                               iaa.ContrastNormalization( r )
                                             )
                              )
       
   
       # Assign sequence to class
       self._seq_imgaug = iaa.Sequential( seq_imgaug )             
    
    

    # ===================================
    # Cast parameter 
    # ===================================
    
    def _cast_parameter( self , entry , format='int' ):
       if entry is not None and entry != 'None':
          if format == 'int':
             return myint( myfloat( entry ) )    

          elif format == 'float':
             return myfloat( entry )

          elif format == 'tuple':
             chunks = entry.split( ':' )
             return ( myfloat( chunks[0] ) , myfloat( chunks[1] ) )
 
          elif format == 'tuple_double':
             chunks  = entry.split( ';' )
             tuple_x = ( myfloat( chunks[0].split( ':' )[0] ) , myfloat( chunks[0].split( ':' )[1] ) )
             tuple_y = ( myfloat( chunks[1].split( ':' )[0] ) , myfloat( chunks[1].split( ':' )[1] ) )
             return { "x": tuple_x , "y": tuple_y } 

          elif format == 'boolean':
             return entry

          elif format == 'string':
             return entry

       else:
          return None



    # ===================================
    # Augment single
    # ===================================

    def _augment_image( self , ind_class , ind_sample ):
       print( 'Doing image n.' , ind_sample )
       # Pick a random index
       rand_int = myint( np.random.choice( self._num_imgs_class[ind_class] , 1 ) )
   
       
       # Load image and mask\
       filein      = self._list_imgs_class[ind_class][rand_int]
       pathout , _ = os.path.split( os.path.abspath( filein ) )
       basename    = os.path.basename( filein )        
       img         = self._load_img( filein , self._ext_in )

     
       # Perform the augmentation
       img_augm  = self._seq_imgaug.augment_image( img )
          

       # Save augmented image and mask
       self._save_image( pathout , basename , img_augm )
   
   
  
    # ===================================
    # Load image 
    # ===================================
   
    def _load_img( self , file_img , ext ):
       # Case A: .npy file
       if ext == '.npy':
          img = np.load( file_img )

  
       # Case B: any other format
       else:
          img = io.imread( file_img )   

       return img



    # ===================================
    # Save augmented image
    # ===================================

    def _save_image( self , pathout , basename , img ):
       # Create output filenames
       fileout = self._create_output_filename( pathout )

    
       # Case A: output extension is .npy 
       if self._ext_out == '.npy':
          np.save( fileout , img )
           
     
       # Case B: output extension is anything else
       else:
          io.imsave( fileout , img )
      
   
   
    # ===================================
    # Create output filename
    # ===================================

    def _create_output_filename( self , pathout , basename ):
       hash = str( random.getrandbits( 128 ) )
       
       fileout  = os.path.join( pathout , basename + '_' + hash + self._ext_out )

       return fileout



    # ===================================
    # Create output table
    # ===================================

    def _create_output_table( self ):    
       # Create output filenames
       fileout = os.path.join( self._pathout_table , 'cnn_table_data_augm.csv' )
   
   
       # Collect all images and masks
       list_imgs = []
       labels    = []

       for i in range( self._num_classes ):
           list_found = sorted( glob.glob( self._path_class[i] + '*' + self._ext_out ) )
           list_imgs += list_found
           labels    += list( np.ones( len( list_found ) ) * i )


       # Write down table
       dict = { 'augm. images': np.array( list_imgs ) ,
                'labels': np.array( labels ) }

       df = pd.DataFrame( dict , columns=[ 'augm. images' , 'labels' ] )
       df.to_csv( fileout , sep=';' , index=False )
       print( '\nWritten output table in: ', fileout )       
    

    
    
# =============================================================================
# Main
# =============================================================================

def main():
    # Get input arguments
    time1 = time.time()
    args = _get_args()
    print( '\n\nDATA AUGMENTATION FOR CNN TRAINING\n' )
    
    
    # Initiate class U-Net to create patches
    augm = DataAugm( args.path_in , 
                     args.config_file , 
                     num_cores = args.num_cores ,
                     ext       = args.img_format ,
                     inplace   = args.inplace )
                              
                              
    # Some prints
    print( 'Input training path: ' , augm._path_in )
    print( 'Total number of images: ', augm._num_imgs , ' with extension: ', augm._ext_in )
    print( 'Total number of augmented images to create: ', augm._num_samples )    
    print( 'Number of classes: ', augm._num_classes )
    print( 'Output format: ', augm._ext_out )
    print( 'Number of cores for multiprocessing: ', augm._num_cores )
    
    for i in range( augm._num_classes ):
        print( '\nNumber of images found in class n.', i,': ', augm._num_imgs_class[i] )
        print( 'Class weight of class n.', i,' is: ', augm._class_weights[i] )
        print( 'Number of samples to create for class n.' , i,' is: ', augm._num_samples_class[i] )

    print( '\nTransformations:' )
    for i in range( len( augm._transform_names ) ):
        print( augm._transform_names[i].upper() , ' ', augm._transform_params[i] )
       
    
    # Perform data augmentation
    print( '\nPerforming data augmentation ....' )
    augm._do_augment()
    print( '.... done!' )
    
    
    # Create output final table
    print( '\nCreating output CSV table ....' )
    augm._create_output_table()


    # Final print
    diff = time.time() - time1  
    print( '\nTime elapsed: ', diff,'\n\n' )
    
    
    
    
# =============================================================================
# Call to main
# =============================================================================

if __name__ == '__main__':
    main()