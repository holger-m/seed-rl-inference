#!/usr/bin/env python

import numpy as np
import pygame
import time

key_action_tform_table = (0,2,5,2,4,7,9,7,3,6,8,6,3,6,8,6,1,10,13,
                          10,12,15,17,15,11,14,16,14,11,14,16,14)

#pygame.init()
pygame.display.init()
pygame.font.init()
# pygame.mixer.init()   # disable sound
#screen = pygame.display.set_mode((800,600), pygame.FULLSCREEN)

screen = pygame.display.set_mode((800,600))
game_surface = pygame.Surface((160,210))
pygame.mouse.set_visible(False)
pygame.display.flip()
clock = pygame.time.Clock()

n_frames = 600
a = 0
responses_vec = np.zeros(n_frames, dtype=np.uint8)
trigger_vec = np.zeros(n_frames, dtype=np.int32)
time_vec = np.zeros(n_frames, dtype=np.int64)
time_diff_vec = np.zeros(n_frames, dtype=np.double)

trigger_count = 0

for loop_count in range(n_frames):
    
    curr_time = time.time_ns()
    time_vec[loop_count] = curr_time
    
    #get the keys
    
    keys = 0
    pressed = pygame.key.get_pressed()
    keys |= pressed[pygame.K_u]
    keys |= pressed[pygame.K_2]  <<1
    keys |= pressed[pygame.K_3]  <<2
    keys |= pressed[pygame.K_4] <<3
    keys |= pressed[pygame.K_1] <<4
    a = key_action_tform_table[keys]
    
    if (pressed[pygame.K_t]):
        trigger_count  += 1
    
    responses_vec[loop_count] = a
    trigger_vec[loop_count] = trigger_count
    
    #clear screen
    screen.fill((0,0,0))
    
    #get atari screen pixels and blit them
    numpy_surface = np.frombuffer(game_surface.get_buffer(),dtype=np.uint8)
    del numpy_surface
    
    screen.blit(pygame.transform.scale(game_surface, (800,600)),(0,0))
    
    pygame.display.flip()
    
    #process pygame event queue
    exit=False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit=True
            break;
    
    if(pressed[pygame.K_q]):
        exit = True
    if(exit):
        break
    
    #delay to 60fps
    clock.tick(60.)


for loop_count in range(n_frames):
    
    if loop_count == 0:
        pass
    else:
        time_diff_vec[loop_count] = (time_vec[loop_count] - time_vec[loop_count-1])/1000000.0  # time in milliseconds

np.savetxt("/workspace/container_mount/ramdisk/responses_vec_.csv", responses_vec, delimiter=",", fmt='%01.0u')
np.savetxt("/workspace/container_mount/ramdisk/trigger_vec_.csv", trigger_vec, delimiter=",", fmt='%01.0u')
np.savetxt("/workspace/container_mount/ramdisk/time_vec_.csv", time_vec, delimiter=",", fmt='%01.0u')
np.savetxt("/workspace/container_mount/ramdisk/time_diff_vec_.csv", time_diff_vec, delimiter=",", fmt='%01.16f')
