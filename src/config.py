class Config:
    IMAGE_SIZE = 256         
    IN_CHANNELS = 3
    OUT_CHANNELS = 3
  
    NGF = 64   # generator feature maps 
    NDF = 64   # discriminator feature maps
  
    PATCH_SIZE = 70   
    LAMBDA_L1 = 100   
    USE_DROPOUT = True   
  
    DEVICE = "cuda"  
