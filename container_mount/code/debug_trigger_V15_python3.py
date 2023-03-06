import numpy as np
import pygame
import time

pygame.display.init()

pygame.display.gl_set_attribute(pygame.GL_ACCELERATED_VISUAL, 0)

print(' ')
print('pygame.display.mode_ok()')
print(pygame.display.mode_ok((800,600), pygame.OPENGL))

screen = pygame.display.set_mode((800,600), pygame.OPENGL)
#screen = pygame.display.set_mode((800,600))

print(' ')
print('pygame.display.Info()')
print(pygame.display.Info())
print(' ')
print('pygame.display.get_driver()')
print(pygame.display.get_driver())
print(' ')
print('pygame.display.get_wm_info(')
print(pygame.display.get_wm_info())
print(' ')
print('pygame.display.get_desktop_sizes()')
print(pygame.display.get_desktop_sizes())
print(' ')
print('pygame.display.list_modes()')
print(pygame.display.list_modes())
print(' ')
print('pygame.display.gl_get_attribute()')
print(' ')
print('GL_MULTISAMPLEBUFFERS')
print(pygame.display.gl_get_attribute(pygame.GL_MULTISAMPLEBUFFERS))
print(' ')
print('GL_STENCIL_SIZE')
print(pygame.display.gl_get_attribute(pygame.GL_STENCIL_SIZE))  # results in an error
# pygame.error: OpenGL error: 00000502
print(' ')
print('GL_DEPTH_SIZE')
print(pygame.display.gl_get_attribute(pygame.GL_DEPTH_SIZE))
print(' ')
print('GL_STEREO')
print(pygame.display.gl_get_attribute(pygame.GL_STEREO))
print(' ')
print('GL_BUFFER_SIZE')
print(pygame.display.gl_get_attribute(pygame.GL_BUFFER_SIZE))
print(' ')
print('GL_CONTEXT_PROFILE_MASK')
print(pygame.display.gl_get_attribute(pygame.GL_CONTEXT_PROFILE_MASK))
print(' ')
print('GL_ACCELERATED_VISUAL')
print(pygame.display.gl_get_attribute(pygame.GL_ACCELERATED_VISUAL))
print(' ')
print('GL_ALPHA_SIZE')
print(pygame.display.gl_get_attribute(pygame.GL_ALPHA_SIZE))
print(' ')

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
