export FWD_ITER=2
export BWD_ITER=1
export DIFF=0.0001
export ACT=relu

LEVELS=2
PROCS=2
BATCH_SIZE=200

mpirun -n ${PROCS} python ../main_lp.py --steps 16 --lp-fwd-cfactor 4 --lp-bwd-cfactor 4 --epochs=2 --seed 2069923971 --lp-fwd-iters ${FWD_ITER} --lp-fwd-levels ${LEVELS} --lp-bwd-levels ${LEVELS} --lp-iters ${BWD_ITER} --batch-size ${BATCH_SIZE} --log-interval 1 --diff-scale ${DIFF} --activation ${ACT} --channels 64 --lp-print 0 --tf 5.0 --lr 0.1 --samp-ratio 0.1 --lp-braid-print 0 | tee results-p${PROCS}.log
