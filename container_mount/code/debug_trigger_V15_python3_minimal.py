import numpy as np
import pygame
import time

pygame.display.init()
screen = pygame.display.set_mode((800,600))
clock = pygame.time.Clock()

n_frames = 600
time_vec = np.zeros(n_frames, dtype=np.int64)
time_diff_vec = np.zeros(n_frames, dtype=np.double)

for i in range(n_frames):
    time_vec[i] = time.time_ns()
    pygame.event.pump()
    pygame.display.flip()
    clock.tick(60.)

print(' ')
for i in range(n_frames):
    if i > 0:
        time_diff_vec[i] = (time_vec[i] - time_vec[i-1])/1000000.0  # time in milliseconds
    if time_diff_vec[i] > 20.0:
        print(str(i+1))

np.savetxt("/workspace/container_mount/ramdisk/time_vec_.csv", time_vec, delimiter=",", fmt='%01.0u')
np.savetxt("/workspace/container_mount/ramdisk/time_diff_vec_.csv", time_diff_vec, delimiter=",", fmt='%01.16f')
