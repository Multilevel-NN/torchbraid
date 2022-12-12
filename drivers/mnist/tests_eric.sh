#!/bin/bash

# ./tests.sh 2>&1 >> output.out

source ~/.bashrc

export PYTHONPATH="/home/jbschroder/joint_repos/torchbraid_holder/torchbraid/torchbraid"
export PYTHONPATH="/home/jbschroder/joint_repos/torchbraid_holder/torchbraid:$PYTHONPATH"
export PYTHONPATH="/home/jbschroder/.local/lib/python3.6/site-packages:$PYTHONPATH"

BATCH_SIZE=50
CHANNELS=8

#NLayers:  
#          np = 2|4   nL = 8
#          np = 3|6   nL = 12
#          np = 4|8   nL = 16
#          np = 6|12  nL = 24

#CG Solve strategy:   
#
#                    Serial with np = 1
#                    1 Level relax, with --lp-finefcf  turned off  ,  cfactor = 2
#                    2 Level relax, with --lp-finefcf  turned off  ,  cfactor = 2
#                    Standard MGRIT with cfactor = 4

########
STEPS=8
echo "nL = $STEPS,  serial"
mpirun -n 1 python3 main_eric.py --steps ${STEPS} --channels ${CHANNELS} --batch-size ${BATCH_SIZE} --log-interval 100 --epochs 20 
#
echo "nL = $STEPS, 1 Level without FCF, cfactor = 2"                                                            
mpirun -n 4 python3 main_eric.py --steps ${STEPS} --channels ${CHANNELS} --batch-size ${BATCH_SIZE} --log-interval 100 --epochs 20 --lp-levels 1  --lp-iters 1 --force-lp --lp-use-relaxonlycg --lp-cfactor 2
#
echo "nL = $STEPS, 2 Level without FCF, cfactor = 2"                                                            
mpirun -n 4 python3 main_eric.py --steps ${STEPS} --channels ${CHANNELS} --batch-size ${BATCH_SIZE} --log-interval 100 --epochs 20 --lp-levels 2  --lp-iters 1 --force-lp --lp-use-relaxonlycg --lp-cfactor 2
#
echo "nL = $STEPS, Standard MGRIT, cfactor = 4"                                                            
mpirun -n 2 python3 main_eric.py --steps ${STEPS} --channels ${CHANNELS} --batch-size ${BATCH_SIZE} --log-interval 100 --epochs 20 --lp-levels 16  --lp-iters 1 --force-lp --lp-cfactor 4

########
STEPS=12
echo "nL = $STEPS,  serial"
mpirun -n 1 python3 main_eric.py --steps ${STEPS} --channels ${CHANNELS} --batch-size ${BATCH_SIZE} --log-interval 100 --epochs 20 
#
echo "nL = $STEPS, 1 Level without FCF, cfactor = 2"                                                            
mpirun -n 6 python3 main_eric.py --steps ${STEPS} --channels ${CHANNELS} --batch-size ${BATCH_SIZE} --log-interval 100 --epochs 20 --lp-levels 1  --lp-iters 1 --force-lp --lp-use-relaxonlycg --lp-cfactor 2
#
echo "nL = $STEPS, 2 Level without FCF, cfactor = 2"                                                            
mpirun -n 6 python3 main_eric.py --steps ${STEPS} --channels ${CHANNELS} --batch-size ${BATCH_SIZE} --log-interval 100 --epochs 20 --lp-levels 2  --lp-iters 1 --force-lp --lp-use-relaxonlycg --lp-cfactor 2
#
echo "nL = $STEPS, Standard MGRIT, cfactor = 4"                                                            
mpirun -n 3 python3 main_eric.py --steps ${STEPS} --channels ${CHANNELS} --batch-size ${BATCH_SIZE} --log-interval 100 --epochs 20 --lp-levels 16  --lp-iters 1 --force-lp --lp-cfactor 4

########
STEPS=16
echo "nL = $STEPS,  serial"
mpirun -n 1 python3 main_eric.py --steps ${STEPS} --channels ${CHANNELS} --batch-size ${BATCH_SIZE} --log-interval 100 --epochs 20 
#
echo "nL = $STEPS, 1 Level without FCF, cfactor = 2"                                                            
mpirun -n 8 python3 main_eric.py --steps ${STEPS} --channels ${CHANNELS} --batch-size ${BATCH_SIZE} --log-interval 100 --epochs 20 --lp-levels 1  --lp-iters 1 --force-lp --lp-use-relaxonlycg --lp-cfactor 2
#
echo "nL = $STEPS, 2 Level without FCF, cfactor = 2"                                                            
mpirun -n 8 python3 main_eric.py --steps ${STEPS} --channels ${CHANNELS} --batch-size ${BATCH_SIZE} --log-interval 100 --epochs 20 --lp-levels 2  --lp-iters 1 --force-lp --lp-use-relaxonlycg --lp-cfactor 2
#
echo "nL = $STEPS, Standard MGRIT, cfactor = 4"                                                            
mpirun -n 4 python3 main_eric.py --steps ${STEPS} --channels ${CHANNELS} --batch-size ${BATCH_SIZE} --log-interval 100 --epochs 20 --lp-levels 16  --lp-iters 1 --force-lp --lp-cfactor 4

########
STEPS=24
echo "nL = $STEPS,  serial"
mpirun -n 1 python3 main_eric.py --steps ${STEPS} --channels ${CHANNELS} --batch-size ${BATCH_SIZE} --log-interval 100 --epochs 20 
#
echo "nL = $STEPS, 1 Level without FCF, cfactor = 2"                                                            
mpirun -n 12 python3 main_eric.py --steps ${STEPS} --channels ${CHANNELS} --batch-size ${BATCH_SIZE} --log-interval 100 --epochs 20 --lp-levels 1  --lp-iters 1 --force-lp --lp-use-relaxonlycg --lp-cfactor 2
#
echo "nL = $STEPS, 2 Level without FCF, cfactor = 2"                                                            
mpirun -n 12 python3 main_eric.py --steps ${STEPS} --channels ${CHANNELS} --batch-size ${BATCH_SIZE} --log-interval 100 --epochs 20 --lp-levels 2  --lp-iters 1 --force-lp --lp-use-relaxonlycg --lp-cfactor 2
#
echo "nL = $STEPS, Standard MGRIT, cfactor = 4"                                                            
mpirun -n 6 python3 main_eric.py --steps ${STEPS} --channels ${CHANNELS} --batch-size ${BATCH_SIZE} --log-interval 100 --epochs 20 --lp-levels 16  --lp-iters 1 --force-lp --lp-cfactor 4





