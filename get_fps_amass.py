import numpy as np

src = '/home/belca/Downloads/02_04_poses.npz'
src = '/home/belca/Downloads/SSM/20160330_03333/shake_hips_stageii.npz'

bdata = np.load(src, allow_pickle=True)

print(bdata.files)

fps = bdata['mocap_frame_rate']
frame_number = bdata['trans'].shape[0]

print(fps)
print(frame_number)