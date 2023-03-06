#!/usr/bin/env python3

import pygame
import time
import socket

def conn_func(sock, clock):
    
    sock.listen()
    print('Waiting for connection...')
    
    conn, addr = sock.accept()
    
    with conn:
        
        print('Connection established with ')
        print('conn:')
        print(conn)
        print('addr:')
        print(addr)
        
        print(' ')
        print('To exit, click into the pygame window and press q')
        print(' ')
        
        prev_time_ns = time.time_ns()
        
        trigger_flag = False
        send_byte = b't0'
        
        exit_conn_func = False
        
        while True:
            
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_t:
                    
                    curr_time_ns = time.time_ns()
                    diff_time_ms = (curr_time_ns - prev_time_ns)/1000000.0
                    prev_time_ns = curr_time_ns
                    
                    trigger_flag = not trigger_flag
                    
                    if trigger_flag:
                        send_byte = b't1'
                    else:
                        send_byte = b't0'
                    
                    print('Got t, time diff is ' + str(round(diff_time_ms,2)) + ' ms')
                    
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    exit_conn_func = True
                    return exit_conn_func
            
            try:
                conn.sendall(send_byte)
            except ConnectionResetError:
                return
            
            #pygame.display.flip()
            #clock.tick_busy_loop(1000)
            clock.tick(1000)

def main():
    
    pygame.display.init()
    pygame.display.set_mode((640,480))
    clock = pygame.time.Clock()
    
    HOST = ''
    PORT = 17171
    
    exit_conn_func = False
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:  # TCP
    #with = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:  # UDP
        
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((HOST, PORT))
        while not exit_conn_func:
            exit_conn_func = conn_func(sock, clock)
        sock.shutdown(socket.SHUT_RDWR)

if __name__ == '__main__':
    main()

