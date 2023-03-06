#!/usr/bin/env python

# ale_python_test_pygame_player.py
# Author: Ben Goodrich
#
# This modified ale_python_test_pygame.py to provide a fully interactive experience allowing the player
# to play. RAM Contents, current action, and reward are also displayed.
# keys are:
# arrow keys -> up/down/left/right
# z -> fire button
import sys
from ale_python_interface import ALEInterface
import numpy as np
import pygame
import time
import fcntl


np.set_printoptions(threshold=sys.maxsize) # print arrays completely

key_action_tform_table = (
0, #00000 none
2, #00001 up
5, #00010 down
2, #00011 up/down (invalid)
4, #00100 left
7, #00101 up/left
9, #00110 down/left
7, #00111 up/down/left (invalid)
3, #01000 right
6, #01001 up/right
8, #01010 down/right
6, #01011 up/down/right (invalid)
3, #01100 left/right (invalid)
6, #01101 left/right/up (invalid)
8, #01110 left/right/down (invalid)
6, #01111 up/down/left/right (invalid)
1, #10000 fire
10, #10001 fire up
13, #10010 fire down
10, #10011 fire up/down (invalid)
12, #10100 fire left
15, #10101 fire up/left
17, #10110 fire down/left
15, #10111 fire up/down/left (invalid)
11, #11000 fire right
14, #11001 fire up/right
16, #11010 fire down/right
14, #11011 fire up/down/right (invalid)
11, #11100 fire left/right (invalid)
14, #11101 fire left/right/up (invalid)
16, #11110 fire left/right/down (invalid)
14  #11111 fire up/down/left/right (invalid)
)

if(len(sys.argv) < 2):
    print("Usage ./pygame_master_Vxx.py <ROM_FILE_NAME>")
    sys.exit()

if(len(sys.argv) < 3):
    print("Usage ./pygame_master_Vxx.py <ROM_FILE_NAME> <SESSION_NO>")
    sys.exit()

save_str = sys.argv[2]
print("Session-No.: " + save_str)

session_exist_flag = False

try:

    test_csv = np.loadtxt(open("/workspace/container_mount/ramdisk/responses_vec_" + save_str + ".csv"))
    print("Session-No. already exists! Exiting...")
    session_exist_flag = True

except:

    print("Session-No. checked: New session starting!")

if (session_exist_flag):

    sys.exit()

game_str = sys.argv[1]

if (game_str.find("breakout") == -1):

    breakout_flag = False

else:

    breakout_flag = True

if (breakout_flag):

    print("Detected breakout, slowing responses!")

else:

    print("Breakout not detected!")

ale = ALEInterface()

max_frames_per_episode = ale.getInt(b"max_num_frames_per_episode");

np.random.seed()  # initialize numpy to random seed via system clock
ale_seed = np.random.randint(2^32)
ale.setInt(b"random_seed",ale_seed)

random_seed = ale.getInt(b"random_seed")
print("random_seed: " + str(random_seed))

ale.setFloat(b'repeat_action_probability', 0.0)
action_repeat_prob = ale.getFloat(b'repeat_action_probability')
print('action repeat prob.: ' + str(action_repeat_prob))

ale.loadROM(str.encode(sys.argv[1]))
legal_actions = ale.getMinimalActionSet()
print(legal_actions)

(screen_width,screen_height) = ale.getScreenDims()
print("width/height: " +str(screen_width) + "/" + str(screen_height))

#(display_width,display_height) = (1280,840)
(display_width,display_height) = (800,600)

#init pygame
#pygame.init()
pygame.display.init()
pygame.font.init()
# pygame.mixer.init()   # disable sound
#screen = pygame.display.set_mode((display_width,display_height), pygame.FULLSCREEN)
screen = pygame.display.set_mode((display_width,display_height))
pygame.display.set_caption("Arcade Learning Environment Player Agent Display")

game_surface = pygame.Surface((screen_width,screen_height))
game_surface_gray = pygame.Surface((screen_width,screen_height))

pygame.mouse.set_visible(False)

pygame.display.flip()

#init clock
clock = pygame.time.Clock()

episode = 0
total_reward = 0.0
total_total_reward = 0.0

n_frames = 108000 # must be divisible by 4
human_play_flag = False

loop_count = 0
loop_count_intro = 0
loop_15hz_count = 0
mod_4_count = 0
a = 0
a_old = 0

rng = np.random.default_rng()
noop = rng.integers(30)
eps_greedy = 0.001

response_flag_0 = np.zeros((1, 1), dtype=np.uint8)
screen_flag_1 = np.ones((1, 1), dtype=np.uint8)
screen_temp = np.zeros((210, 160, 3), dtype=np.uint8)
screen_temp_gray = np.zeros((210, 160), dtype=np.uint8)
episode_end_vec = np.zeros((4, 1), dtype=np.uint8)
episode_end_flag = np.zeros((1, 1), dtype=np.uint8)

#screen_15hz_RGB = np.zeros((int(n_frames/4), 2*210*160), dtype=np.int32)
screen_15hz_RGB = np.zeros((210, 160, 3, 2), dtype=np.uint8)
screen_15hz_Gray = np.zeros((210, 160, 2), dtype=np.uint8)
#responses_vec = np.zeros(n_frames, dtype=np.uint8)
reward_vec = np.zeros(n_frames, dtype=np.double)
episode_vec = np.zeros(n_frames, dtype=np.int32)
#trigger_vec = np.zeros(n_frames, dtype=np.int32)
#trigger_vec_intro = np.zeros(18000, dtype=np.int32)  # 5 min

waiting_for_trigger_flag = True
trigger_count = 0

if (human_play_flag):

    while(waiting_for_trigger_flag):

        pressed = pygame.key.get_pressed()

        if (pressed[pygame.K_t]):

            trigger_count += 1

        trigger_vec_intro[loop_count_intro] = trigger_count

        screen.fill((0,0,0))

        font = pygame.font.SysFont("Ubuntu Mono",32)
        text = font.render("Session-No.: " + save_str, 1, (255,255,255))
        screen.blit(text,(280,110))
        text = font.render("Trigger-No.: " + str(trigger_count), 1, (255,255,255))
        screen.blit(text,(280,210))

        if (trigger_count == 11):

            waiting_for_trigger_flag = False

        pygame.display.flip()

        exit=False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit=True
                break;

        if(pressed[pygame.K_q]):
            exit = True

        if(exit):
            break

        clock.tick(60.)

        loop_count_intro += 1

if (not human_play_flag):
    time_start = time.time()

while(loop_count < n_frames):
#while(episode < 10):

    mod_4_count += 1

    #get the keys

    if (human_play_flag):

        keys = 0
        pressed = pygame.key.get_pressed()
        #keys |= pressed[pygame.K_UP]
        #keys |= pressed[pygame.K_DOWN]  <<1
        #keys |= pressed[pygame.K_LEFT]  <<2
        #keys |= pressed[pygame.K_RIGHT] <<3
        #keys |= pressed[pygame.K_z] <<4
        keys |= pressed[pygame.K_u]
        keys |= pressed[pygame.K_2]  <<1
        keys |= pressed[pygame.K_3]  <<2
        keys |= pressed[pygame.K_4] <<3
        keys |= pressed[pygame.K_1] <<4
        a = key_action_tform_table[keys]

        if (pressed[pygame.K_t]):

            trigger_count  += 1


    if (human_play_flag and breakout_flag):

        if (a_old != 0):

            a = 0
            
        a_old = a
    
    #TODO: BUG! noop must have extra counter starting from 0 when new episode begins
    if loop_count <= noop:
        a = 0
    
    u_greedy = rng.random()
    if u_greedy <= eps_greedy:
        a = rng.integers(18)
    
    #responses_vec[loop_count] = a
    reward = ale.act(a);
    total_reward += reward
    total_total_reward += reward

    reward_vec[loop_count] = total_reward
    episode_vec[loop_count] = episode
    #trigger_vec[loop_count] = trigger_count

    #clear screen
    screen.fill((0,0,0))

    #get atari screen pixels and blit them
    numpy_surface = np.frombuffer(game_surface.get_buffer(),dtype=np.uint8)
    numpy_surface_gray = np.frombuffer(game_surface_gray.get_buffer(),dtype=np.uint8)
    ale.getScreenRGB(screen_temp)
    ale.getScreenGrayscale(screen_temp_gray)
    screen_temp_reversed = np.reshape(np.dstack((screen_temp[:,:,2], screen_temp[:,:,1], screen_temp[:,:,0], np.zeros((screen_height, screen_width), dtype=np.uint8))), 210*160*4)
    screen_temp_reversed_gray = np.reshape(np.dstack((screen_temp_gray, screen_temp_gray, screen_temp_gray, np.zeros((screen_height, screen_width), dtype=np.uint8))), 210*160*4)
    numpy_surface[:] = screen_temp_reversed
    numpy_surface_gray[:] = screen_temp_reversed_gray
    
    episode_end_vec[mod_4_count - 1] = 0
    
    if(ale.game_over()):
        episode_end_vec[mod_4_count - 1] = 1
    
    if(mod_4_count == 3):
        
        screen_15hz_RGB[:,:,:,0] = screen_temp
        screen_15hz_Gray[:,:,0] = screen_temp_gray

    if(mod_4_count == 4):

        mod_4_count = 0        

        screen_15hz_RGB[:,:,:,1] = screen_temp
        screen_15hz_Gray[:,:,1] = screen_temp_gray
        
        if np.sum(episode_end_vec) == 0:
            episode_end_flag[0] = 0
        else:
            episode_end_flag[0] = 1

        if (not human_play_flag):

            #last_15hz_screen = np.reshape(screen_15hz_RGB[:,:,:,loop_15hz_count], (1,67200))
            last_15hz_screen = screen_15hz_RGB
            last_15hz_screen_gray = screen_15hz_Gray

            #np.savetxt("/media/ramdisk/last_15hz_screen.csv", last_15hz_screen, delimiter=",", fmt='%01.0u')
            np.save('/workspace/container_mount/ramdisk/last_15hz_screen.npy', last_15hz_screen_gray)
            np.save('/workspace/container_mount/ramdisk/episode_end_flag.npy', episode_end_flag)
            fd_screen = open('/workspace/container_mount/ramdisk/screenlock', 'r')
            fcntl.flock(fd_screen, fcntl.LOCK_EX)
            np.savetxt("/workspace/container_mount/ramdisk/screen_flag.csv", screen_flag_1, fmt='%01.0u')
            fcntl.flock(fd_screen, fcntl.LOCK_UN)
            fd_screen.close()

            repeat_wait_flag = True

            while repeat_wait_flag:
                fd_response = open('/workspace/container_mount/ramdisk/responselock', 'r')
                fcntl.flock(fd_response, fcntl.LOCK_EX)
                response_flag_csv = np.loadtxt(open("/workspace/container_mount/ramdisk/response_flag.csv"))
                fcntl.flock(fd_response, fcntl.LOCK_UN)
                fd_response.close()
                if (response_flag_csv == 1):
                    time_end = time.time()
                    #print(loop_count)
                    #print(time_end - time_start)
                    #print(" ")
                    time_start = time.time()
                    np.savetxt("/workspace/container_mount/ramdisk/response_flag.csv", response_flag_0, fmt='%01.0u')
                    a = np.loadtxt(open("/workspace/container_mount/ramdisk/action_ind_ALE.csv"))
                    repeat_wait_flag = False
                else:
                    time.sleep(0.01)

        loop_15hz_count += 1



    del numpy_surface
    del numpy_surface_gray
    #screen.blit(pygame.transform.scale2x(game_surface),(0,0))
    screen.blit(pygame.transform.scale(game_surface, (display_width,display_height)),(0,0))

    #font = pygame.font.SysFont("Ubuntu Mono",32)
    #text = font.render("Trigger-No.: " + str(trigger_count), 1, (255,255,255))
    #screen.blit(text,(330,210))

    #get RAM
    #ram_size = ale.getRAMSize()
    #ram = np.zeros((ram_size),dtype=np.uint8)
    #ale.getRAM(ram)


    #Display ram bytes
    #font = pygame.font.SysFont("Ubuntu Mono",32)
    #text = font.render("RAM: " ,1,(255,208,208))
    #screen.blit(text,(330,10))

    #font = pygame.font.SysFont("Ubuntu Mono",25)
    #height = font.get_height()*1.2

    #line_pos = 40
    #ram_pos = 0
    #while(ram_pos < 128):
        #ram_string = ''.join(["%02X "%ram[x] for x in range(ram_pos,min(ram_pos+16,128))])
        #text = font.render(ram_string,1,(255,255,255))
        #screen.blit(text,(340,line_pos))
        #line_pos += height
        #ram_pos +=16
        
    #display current action
    #font = pygame.font.SysFont("Ubuntu Mono",32)
    #text = font.render("Current Action: " + str(a) ,1,(208,208,255))
    #height = font.get_height()*1.2
    #screen.blit(text,(330,line_pos))
    #line_pos += height

    #display reward
    #font = pygame.font.SysFont("Ubuntu Mono",30)
    #text = font.render("Total Reward: " + str(total_reward) ,1,(208,255,255))
    #screen.blit(text,(330,line_pos))

    pygame.display.flip()

    #process pygame event queue
    exit=False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit=True
            break;

    if(human_play_flag):
        if(pressed[pygame.K_q]):
            exit = True
    if(exit):
        break

    #delay to 60fps
    clock.tick(60.)
    
    if (loop_count + 1)%1000 == 0:
        episode_frame_number = ale.getEpisodeFrameNumber()
        frame_number = ale.getFrameNumber()
        print("Total Frame Number: " + str(loop_count))
        print("Total Frame Number ALE: " + str(frame_number))
        print("Episode Number: " + str(episode))
        print("Episode Frame Number ALE: " + str(episode_frame_number))
        print("Episode Score: " + str(total_reward))
        print(" ")
    
    if(ale.game_over()):
        episode_frame_number = ale.getEpisodeFrameNumber()
        frame_number = ale.getFrameNumber()
        print("EPISODE END!")
        print("Total Frame Number: " + str(loop_count))
        print("Total Frame Number ALE: " + str(frame_number))
        print("Episode Number: " + str(episode))
        print("Episode Frame Number ALE: " + str(episode_frame_number))
        print("Episode Score: " + str(total_reward))
        print(" ")
        ale.reset_game()
        total_reward = 0.0
        episode = episode + 1

    loop_count += 1

result_count = 0

while(result_count < 120):

    screen.fill((0,0,0))

    font = pygame.font.SysFont("Ubuntu Mono",32)
    text = font.render("Block " + save_str + " completed!", 1, (255,255,255))
    screen.blit(text,(260,210))
    text = font.render("Total score: " + str(total_total_reward), 1, (255,255,255))
    height = font.get_height()*1.2
    screen.blit(text,(260,210 + height))

    pygame.display.flip()

    #process pygame event queue
    exit=False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit=True
            break;

    if(human_play_flag):
        if(pressed[pygame.K_q]):
            exit = True
    if(exit):
        break

    clock.tick(60.)

    result_count += 1

screen.fill((0,0,0))

font = pygame.font.SysFont("Ubuntu Mono",32)
text = font.render("Done, writing files...", 1, (255,255,255))
screen.blit(text,(260,210))

pygame.display.flip()

pygame.event.pump()

clock.tick(60.)

#np.savetxt("/media/ramdisk/screen_15hz_RGB_" + save_str + ".csv", screen_15hz_RGB, delimiter=",", fmt='%01.0u')
#np.save("/workspace/container_mount/ramdisk/screen_15hz_RGB_" + save_str, screen_15hz_RGB)
#np.savetxt("/workspace/container_mount/ramdisk/responses_vec_" + save_str + ".csv", responses_vec, delimiter=",", fmt='%01.0u')
np.savetxt("/workspace/container_mount/ramdisk/reward_vec_" + save_str + ".csv", reward_vec, delimiter=",", fmt='%01.1f')
np.savetxt("/workspace/container_mount/ramdisk/episode_vec_" + save_str + ".csv", episode_vec, delimiter=",", fmt='%01.0u')
#np.savetxt("/workspace/container_mount/ramdisk/trigger_vec_" + save_str + ".csv", trigger_vec, delimiter=",", fmt='%01.0u')
#np.savetxt("/workspace/container_mount/ramdisk/trigger_vec_intro_" + save_str + ".csv", trigger_vec_intro, delimiter=",", fmt='%01.0u')

