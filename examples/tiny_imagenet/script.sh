export FWD_ITER=1
export BWD_ITER=1
export DIFF=0.0001
export ACT=relu

LEVELS=1
PROCS=1

mpirun -n ${PROCS} python /Users/eccyr/Packages/torchbraid/examples/tiny_imagenet/main_lp.py --steps 3 --lp-fwd-cfactor 4 --lp-bwd-cfactor 4 --epochs=8 --seed 2069923971 --lp-fwd-iters ${FWD_ITER} --lp-fwd-levels ${LEVELS} --lp-bwd-levels ${LEVELS} --lp-iters ${BWD_ITER} --batch-size 200 --log-interval 5 --diff-scale ${DIFF} --activation ${ACT} --samp-ratio 1.0 --channels 64 --lp-braid-print 0 --lp-print 0 --tf 5.0 --lr 0.1 --samp-ratio 0.1
