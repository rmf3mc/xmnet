def get_folder_path(args):
    
    path_parts = []

    # Append the name of the argument to the list if it's True
    
    
    
    if args.mmanet:
        path_parts.append('MMANet')

    elif args.xmnet:
        path_parts.append('XMNet')
        
    else:
        path_parts.append('Original')
        
        
    if args.maskguided:
        path_parts.append('mask_guided')
              

    if args.cls_included:
        path_parts.append('_Classification')
        
        
                
    if args.seg_included:
        path_parts.append('_Segmentation')
        if args.Unet:
            path_parts.append('_Unet')
        else:
            path_parts.append('_Unet3Plus')
  



        

    # Join the parts together with underscores
    args_part = '_'.join(path_parts)
    
    
    return args_part
    
    
    
def iou_binary(preds, labels, EMPTY=1e-9):
    """
    Calculate Intersection over Union (IoU) for a single pair of binary segmentation masks using bitwise operations.
    
    Parameters:
    - preds (Tensor): Predicted segmentation mask, shape [1, height, width]
    - labels (Tensor): Ground truth segmentation mask, shape [1, height, width]
    - EMPTY (float): A small constant to prevent division by zero
    
    Returns:
    - IoU (float): Intersection over Union score
    """
    
    # Convert to boolean tensors
    preds = preds.bool()
    labels = labels.bool()

    # Calculate intersection and union using bitwise operations
    intersection = (preds & labels).float().sum()
    union = (preds | labels).float().sum()

    # Calculate IoU
    IoU = (intersection + EMPTY) / (union + EMPTY)

    return IoU.item()