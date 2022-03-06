export FWD_ITER=8
export BWD_ITER=8
export DIFF=0.0001
export ACT=relu

LEVELS=-1
PROCS=4

mpirun -n ${PROCS} python /Users/eccyr/Packages/torchbraid/examples/tiny_imagenet/main_lp.py --steps 16 --lp-fwd-cfactor 4 --lp-bwd-cfactor 4 --epochs=8 --seed 2069923971 --lp-fwd-iters ${FWD_ITER} --lp-fwd-levels ${LEVELS} --lp-bwd-levels ${LEVELS} --lp-iters ${BWD_ITER} --batch-size 100 --log-interval 5 --diff-scale ${DIFF} --activation ${ACT} --samp-ratio 0.1 --channels 8 --lp-braid-print 0 --lp-print 0 --tf 17.0
