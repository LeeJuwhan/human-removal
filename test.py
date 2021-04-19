from main import main

def inpainting(mask_roi_path, image_roi_path,results_path):
    main(mask_roi_path, image_roi_path,results_path, mode=2)