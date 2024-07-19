#!/usr/bin/env python

'''
Stitching sample
================

This script demonstrates how to use the Stitcher API from OpenCV to stitch
two images into a panorama and display the result.
'''

import numpy as np
import cv2 as cv
import sys

def main(left_img_path, right_img_path, output_path='result.jpg'):
    # Read input images
    left_img = cv.imread(cv.samples.findFile(left_img_path))
    if left_img is None:
        print(f"Can't read image {left_img_path}")
        sys.exit(-1)

    right_img = cv.imread(cv.samples.findFile(right_img_path))
    if right_img is None:
        print(f"Can't read image {right_img_path}")
        sys.exit(-1)

    # Create a Stitcher object
    stitcher = cv.Stitcher.create(cv.Stitcher_PANORAMA)
    status, pano = stitcher.stitch([left_img, right_img])

    if status != cv.Stitcher_OK:
        print(f"Can't stitch images, error code = {status}")
        sys.exit(-1)

    # Save the result
    cv.imwrite(output_path, pano)
    print(f"Stitching completed successfully. {output_path} saved!")

    # Display the result
    cv.imshow('Panorama', pano)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python stitching.py <left_img_path> <right_img_path> [<output_path>]")
        sys.exit(-1)
    
    left_img_path = sys.argv[1]
    right_img_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else 'result.jpg'
    
    main(left_img_path, right_img_path, output_path)
