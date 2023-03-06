# seed-rl-basteln

Important: nvidia docker must be installed in order to make the tf-image work!
See https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

Type "xhost + local:docker" on host to give ale_task_present container display access

Run "set_ramdisk_on_host" before collecting task data

Explanation: the --ipc=host flag in docker run is set to avoid the error message:
Major opcode of failed request: 130 (MIT-SHM)
See also: https://github.com/mviereck/x11docker/blob/master/x11docker (search for mit-shm)

Explanation: the --network="host" flag is set to get a connection to the TCP-based trigger signal

-------------------------------------------------------------

When running the seed_rl container for the first time, copy ROMs directory on host to ~/seedrl_basteln_data/container_mount/data, and then run inside the container:

cd /workspace/container_mount/data/ROMs
unrar e Roms.rar
unzip ROMS.zip
unzip "HC ROMS.zip"
rm ROMS.zip
rm "HC ROMS.zip"
rm Roms.rar

To import ROMs into gym, run inside container:
python -m atari_py.import_roms /workspace/container_mount/data/ROMs

Network weights from Google can be found on the DFG3610 group drive:
smb://vs-grp07.zih.tu-dresden.de/dfg3610/Shared/Atari_Seed_RL

Extracted files must be copied to the host here:
~/seedrl_basteln_data/container_mount/data/Checkpoints_from_google/GameXYZ

In order to make cuda compute capability 8.6 work with tf 2.4.1, we have to replace the file ptxas of cuda 11.0 (located at /usr/local/cuda-11.0/bin/ptxas inside the container) by the ptxas of cuda 11.2 (see here: https://github.com/tensorflow/tensorflow/issues/45590#issuecomment-780678654). A dockerfile for cuda 11.2 is located here: 
~/gitlab_mn/seed-rl-basteln/container_mount/code/cuda_11_2_docker/dockerfile_cuda_11_2

Once copied from the cuda_11_2 container to 
~/seedrl_basteln_data/container_mount/data/cuda_11_2_ptxas_file/ptxas
the file must be copied manually into the seed-rl-basteln container after every rebuild (to /usr/local/cuda-11.0/bin/ptxas). 
