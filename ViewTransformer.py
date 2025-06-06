import numpy as np
import cv2
class ViewTransformer:
    def __init__(self,source:np.ndarray,target:np.ndarray):
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self,points:np.ndarray):
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1,1,2).astype(np.float32)
        transformed_points =  cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1,2)