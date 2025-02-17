torchrun --nproc_per_node=2 tools/train.py -c /home/prml/StudentsWork/Chanyoung/MODU/configs/MODU/atto.yml -r /home/prml/StudentsWork/Chanyoung/MODU/result/atto/checkpoint.pth
torchrun --nproc_per_node=2 tools/train.py -c /home/prml/StudentsWork/Chanyoung/MODU/configs/MODU/femto.yml
torchrun --nproc_per_node=2 tools/train.py -c /home/prml/StudentsWork/Chanyoung/MODU/configs/MODU/pico.yml 