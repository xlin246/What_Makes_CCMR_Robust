import numpy as np
from PIL import Image
from core.utils.flow_IO import readFlowFile
from core.utils.flow_viz import flow_to_image
import os
import sys

'''
Example arguements:
python visualization.py data/sintel/training/final/alley_1/frame_0001.png data/sintel/training/final/alley_1/frame_0001.png data/sintel/training/flow/alley_1/frame_0001.flo flo_img_sintel_alley_1_frame_0001.png
'''

def visualize_flow(image1_path,image2_path,flo_path,save_path):
    image1 = np.array(Image.open(image1_path).convert("RGB"))
    image2 = np.array(Image.open(image2_path).convert("RGB"))

    flow = readFlowFile(flo_path)
    flow_image = flow_to_image(flow)

    h,w = image1.shape[:2]

    image2 = np.array(Image.fromarray(image2).resize((w,h)))
    flow_image = np.array(Image.fromarray(flow_image).resize((w,h)))

    '''
    1.Displayed in one row: 
    result = np.concatenate([image1,image2,flow_image],axis=1) # show image1,image2 and flow image.
    # result = flow_image # only show the flow image without image1 and image2.
    '''
    
    '''
    2.Displayed in two rows:
    '''
    padding = np.ones((h, w, 3), dtype=np.uint8)*255
    row1 = np.concatenate([flow_image, padding], axis=1) 
    row2 = np.concatenate([image1, image2], axis=1)
    result = np.concatenate([row1, row2], axis=0)
    
    os.makedirs(os.path.dirname(save_path),exist_ok =True) if os.path.dirname(save_path) else None
    Image.fromarray(result).save(save_path)

if __name__ == '__main__':
    if len(sys.argv)!=5:
        print("Valid arguments should be visualization.py <image1_path> <image2_path> <flo_path> <save_path>")
        sys.exit(1)
    image1_path = sys.argv[1]
    image2_path = sys.argv[2]
    flo_path = sys.argv[3]
    save_path = sys.argv[4]

    visualize_flow(image1_path,image2_path,flo_path,save_path)
    #print("visualization.py finished.")

