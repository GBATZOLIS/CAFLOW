import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def read_scan(path):
    scan = nib.load(path)
    scan = scan.get_fdata()
    return scan

mask_path='/Users/gbatz97/Desktop/midbrain_mask_2mm.nii'

mask = read_scan(mask_path)
print(mask.shape)

plt.figure()
plt.title('axis 0')
plt.imshow(mask[mask.shape[0]//2, :, 20:35])
plt.savefig('/Users/gbatz97/Desktop/axis0.png')

plt.figure()
plt.title('axis 1')
plt.imshow(mask[:, mask.shape[1]//2, 20:35])
plt.savefig('/Users/gbatz97/Desktop/axis1.png')

plt.figure()
plt.title('axis 2')
plt.imshow(mask[:, :, 27])
plt.savefig('/Users/gbatz97/Desktop/axis2.png')