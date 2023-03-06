#!/usr/bin/env python3

import pygame
import socket
import time

pygame.display.init()
pygame.display.set_mode((640,480))
clock = pygame.time.Clock()
current_trigger_byte = b't0'

main_loop_flag = True
trigger_count = 0

HOST = '169.254.123.17'
PORT = 17171

#with socket.socket(socket.AF_INET, socket.SOCK_STREAM | socket.SOCK_NONBLOCK) as sock:
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    
    sock.connect((HOST, PORT))
    
    """
    try:
        sock.connect((HOST, PORT))
    except BlockingIOError as err:
        print('While connecting, got error:')
        print(err)
    
    time.sleep(1)
    """
    
    sock.setblocking(False)
    time.sleep(1)
    
    prev_time_ns = time.time_ns()
    
    while main_loop_flag:
        
        data = sock.recv(1024)
        
        if data == b'':
            raise BlockingIOError('No data received, lost connection!')
        
        """
        try:
            data = sock.recv(1024)
        except BlockingIOError as err:
            print('While receiving, got error:')
            print(err)
            data = None
        """
        
        print('Received ' + repr(data.decode()))
        
        new_trigger_byte = data[-2:]
        
        if new_trigger_byte != current_trigger_byte:
            
            trigger_count += 1
            current_trigger_byte = new_trigger_byte
            
            curr_time_ns = time.time_ns()
            diff_time_ms = (curr_time_ns - prev_time_ns)/1000000.0
            prev_time_ns = curr_time_ns
            
            print(' ')
            print('time diff is ' + str(round(diff_time_ms,2)) + ' ms')
            print('Trigger count: ' + str(trigger_count))
        
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                main_loop_flag = False
        
        #pygame.display.flip()
        clock.tick(60)
    
    sock.shutdown(socket.SHUT_RDWR)
    #sock.close()
