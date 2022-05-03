export FWD_ITER=1
export BWD_ITER=1
export DIFF=0.0001
export ACT=relu

LEVELS=2
PROCS=4
BATCH_SIZE=200

mpirun -n ${PROCS} python ../main_lp.py --steps 16 --lp-fwd-cfactor 4 --lp-bwd-cfactor 4 --epochs=4 --seed 2069923971 --lp-fwd-iters ${FWD_ITER} --lp-fwd-levels ${LEVELS} --lp-bwd-levels ${LEVELS} --lp-iters ${BWD_ITER} --batch-size ${BATCH_SIZE} --log-interval 1 --diff-scale ${DIFF} --activation ${ACT} --samp-ratio 1.0 --channels 64 --lp-print 0 --tf 5.0 --lr 0.1 --samp-ratio 0.1 --lp-braid-print 1 | tee results.log

# --use-serial    : use a serial only neural network (no braid)
