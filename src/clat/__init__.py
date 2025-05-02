import cv2

# avoid overload of CPU with multiple GPU envs
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
