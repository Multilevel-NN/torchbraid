import numpy as np
import os
import sys


def gen_header(queue, nnodes, time, num_omp_threads):
  ''' Generate and return SBATCH header '''

  Header = "#!/bin/bash\n" +\
           "#SBATCH -p " + queue + "\n" +\
           "#SBATCH -N "+str(nnodes) + "\n" +\
           "#SBATCH -A paratime\n" +\
           "#SBATCH -t " + time + "\n" +\
           "#SBATCH -o test_out\n" +\
           "#SBATCH -e test_err\n\n" +\
           "# >>> conda initialize >>>\n" +\
           "# !! Contents within this block are managed by \'conda init\' !! \n" +\
           "__conda_setup=\"$(\'/usr/bin/conda\' \'shell.bash\' \'hook\' 2> /dev/null)\" \n" +\
           "if [ $? -eq 0 ]; then\n" +\
           "    eval \"$__conda_setup\" \n" +\
           "else\n" +\
           "    if [ -f \"/usr/etc/profile.d/conda.sh\" ]; then\n" +\
           "        . \"/usr/etc/profile.d/conda.sh\" \n" +\
           "    else\n" +\
           "        export PATH=\"/usr/bin:$PATH\" \n" +\
           "    fi\n" +\
           "fi\n" +\
           "unset __conda_setup\n" +\
           "# <<< conda initialize <<<\n\n" +\
           "conda activate torchbraid_2_2022 \n\n" +\
           "export OMP_NUM_THREADS=" + str(num_omp_threads) + "\n\n" +\
           "export PYTHONPATH=\"$HOME/2022_torchbraid/torchbraid/torchbraid\" \n" +\
           "export PYTHONPATH=\"$HOME/2022_torchbraid/torchbraid:$PYTHONPATH\" \n\n"

  return Header


def main():
  '''
  This function generates sbatch scripts to run, for a particular parameter
  setting, Adam, NI, NI+MGOPT V(1,1), NI+MGOPT V(0,1), and NI+MGOPT+LocalRelax,
  
  To run, just type
  $ python3 generate_quartz_script.py
  '''
  
  ##
  # Use of this script is a bit clunky, you have to manually tweak the below run_strings to get everything right 
  #
  # Keeping cost the same via number of epochs 
  #   -- Cost of V(1,1) MGOPT epoch plus NI bootstrapping:
  #      (2 total fine-grid relaxations  + 1 g_h  + 1 coarse-grid relaxation + 1 g_H  +  Line-search cost)  +   NI epochs X 2 Levels
  #      In terms of fine-grid gradient evaluations,         
  #      (2                              + 1      + 0.55                     +  0.55  +  Line-search cost)  +   NI epochs X 1.6
  #      (4.1 + Line-search cost) + NI epochs X 1.6  

  #   -- Cost for simple line-search, letting the cost be about 1 optimizer steps is
  #      166 = 32*(4.1 + 1.0) + 3.2
  #   -- Cost using no line search is equal when doing 40 MGOpt epochs and 2 NI epochs
  #      167 = 40*(4.1 + 0.0) + 3.2
  # 
  #   -- Do only 1 relaxation (post relaxation) and assume no significant line search cost, then you can use 53 epochs for an overall cost
  #      167 = 53*(3.1 + 0.0) + 3.2
  #
  #   -- NI epochs are then
  #      167 epochs for single level
  #      92 epochs for two levels (coarse grid is a little bit cheaper, so allow for a bit more epochs, this value computed based on num parameters) 

  ##
  # Global parameters
  ntests = 1                                     # number of individual training runs to do
  nnodes = int(ntests)                           # total number of nodes for nonparallel runs
  #
  #                                              # Parameters for the layer parallel (LP) runs
  #                                              # You want:  nnodes_per_test * nmpi_tasks_per_node  *  braid_coarsening_factor   =   steps
  nnodes_per_test = 4                            
  nnodes_parallel = int(ntests*nnodes_per_test)  # total number of nodes for parallel runs
  nmpi_tasks_per_node = 4                        # This and cpus per task must equal number of cores on machine, i.e., 9*4 = 36 for quartz
                                                 
  cpus_per_task = 9                              # cores per task, for quartz, we do 4 MPI ranks per node, with 9 threads per task, for all 36 cores
  num_omp_threads = 3                            # number of threads per task, usually lower than cpus per task
  

  preamble = 'srun -N 1 -n 1 --cpus-per-task=' + str(cpus_per_task) + ' --cpu-bind=verbose --mpibind=off python3 '
  preamble_parallel = 'srun -N ' + str(nnodes_per_test) + ' -n ' + str(nnodes_per_test*nmpi_tasks_per_node) + " --ntasks-per-node=" + str(nmpi_tasks_per_node) + " --cpus-per-task=" + str(cpus_per_task) + ' --cpu-bind=verbose --mpibind=off python3 '

  queue = 'pdebug' #'pbatch'
  time = "00:15:00"   #"23:55:00"

  ##
  # Run string parameters
  steps = 128
  channels = 8
  batchsize = 50
  #
  sampratio = 1.00
  preserve_optim = 0
  #
  Adamepochs = 60 #167
  NIepochs = 33 #92
  MGOptepochs = 2 #17 #40
  MGOptLPepochs = 2 #17 #53


  # One-level NI, plain Adam
  Adam_run_string = ' main_mgopt.py --steps ' + str(steps) + ' --channels ' + str(channels) + ' --samp-ratio ' + str(sampratio) + ' --zero-init-guess 0 --mgopt-printlevel 1 --ni-levels 1 --lp-fwd-levels 1 --lp-bwd-levels 1 --mgopt-iter 0 --NIepochs ' + str(Adamepochs) + ' --batch-size ' + str(batchsize)
  TB_Adam_file = 'TB_onelevel_Adam'
  
  # Two-level NI 
  NI_run_string = ' main_mgopt.py --steps ' + str(steps) + ' --channels ' + str(channels) + ' --samp-ratio ' + str(sampratio) + ' --zero-init-guess 0 --mgopt-printlevel 1 --ni-levels 2 --lp-fwd-levels 1 --lp-bwd-levels 1 --mgopt-iter 0 --NIepochs ' + str(NIepochs) + ' --batch-size ' + str(batchsize)
  TB_NI_file = 'TB_NI'

  # Two-level, MGOPT
  NI_MGOpt_run_string = 'main_mgopt.py --steps ' + str(steps) + ' --channels ' + str(channels) + ' --samp-ratio ' + str(sampratio) + ' --zero-init-guess 0 --mgopt-printlevel 1 --ni-levels 2 --mgopt-levels 2 --mgopt-nrelax-pre 0 --mgopt-nrelax-post 1 --mgopt-nrelax-coarse 1 --lp-fwd-levels 1 --lp-bwd-levels 1 --lp-iters 1 --preserve-optim ' + str(preserve_optim) + ' --NIepochs 2 --epochs ' + str(MGOptepochs) + ' --batch-size ' + str(batchsize)
  TB_NI_MGOpt_file = 'TB_NI_MGOpt_twolevel'

  # Two-level, MGOPT + Layer Parallel (LP)
  NI_MGOpt_LP_run_string = 'main_mgopt.py --steps ' + str(steps) + ' --channels ' + str(channels) + ' --samp-ratio ' + str(sampratio) + ' --zero-init-guess 0 --mgopt-printlevel 1 --ni-levels 2 --mgopt-levels 2 --mgopt-nrelax-pre 0 --mgopt-nrelax-post 1 --mgopt-nrelax-coarse 1 --lp-fwd-cfactor 8 --lp-bwd-cfactor 8 --lp-fwd-levels 2 --lp-bwd-levels 2 --lp-iters 2 --lp-fwd-iters 2 --preserve-optim ' + str(preserve_optim) + ' --NIepochs 2 --epochs ' + str(MGOptLPepochs) + ' --batch-size ' + str(batchsize) + ' --lp-fwd-finalrelax '
  TB_NI_MGOpt_LP_file = 'TB_NI_MGOpt_twolevel_LP'

  # Two-level LR, MGOPT + Layer Parallel (LP) + Local Relaxation
  #   Keeping cost here equal is a bit tricky because of the LR
  #   Currently just do LR only on the backward pass...
  NI_MGOpt_LP_LR_run_string = 'main_mgopt.py --steps ' + str(steps) + ' --channels ' + str(channels) + ' --samp-ratio ' + str(sampratio) + ' --zero-init-guess 0 --mgopt-printlevel 1 --ni-levels 2 --mgopt-levels 2 --mgopt-nrelax-pre 0 --mgopt-nrelax-post 1 --mgopt-nrelax-coarse 1 --lp-fwd-cfactor 8 --lp-bwd-cfactor 8 --lp-fwd-levels 2 --lp-bwd-levels 1 --lp-iters 2 --preserve-optim ' + str(preserve_optim) + ' --NIepochs 2 --epochs ' + str(MGOptLPepochs) + ' --batch-size ' + str(batchsize) + " --lp-bwd-finefcf --lp-bwd-relaxonlycg"
  TB_NI_MGOpt_LP_LR_file = 'TB_NI_MGOpt_twolevel_LP_LR'


  ##
  # Construct SBatch header
  #     Used to have this line, caused issues  "#SBATCH -c 36\n" +\
  Header_noparallel = gen_header(queue, nnodes, time, num_omp_threads)
  Header_parallel   = gen_header(queue, nnodes_parallel, time, num_omp_threads)
  Footer = "\nwait \n\n"

  ##
  # Open files, and dump header
  fAdam        = open(TB_Adam_file + '.sbatch', "w")
  fNI          = open(TB_NI_file + '.sbatch', "w")
  fMGOPT       = open(TB_NI_MGOpt_file + '.sbatch', "w")
  fMGOPT_LP    = open(TB_NI_MGOpt_LP_file + '.sbatch', "w")
  fMGOPT_LP_LR = open(TB_NI_MGOpt_LP_LR_file + '.sbatch', "w")
  #
  fAdam.write(Header_noparallel)
  fNI.write(Header_noparallel)
  fMGOPT.write(Header_noparallel)
  fMGOPT_LP.write(Header_parallel)
  fMGOPT_LP_LR.write(Header_parallel)

  ##
  # Dump our tests to each file
  background = ''
  if ntests > 1:
    background = ' &\n'
  for k in range(ntests):
    fAdam.write(preamble + Adam_run_string + ' --seed ' + str(k*11) + '  >  temp_' + TB_Adam_file + '_' + str(k) + '.out' + background) 
    fNI.write(preamble + NI_run_string + ' --seed ' + str(k*11) + '  >  temp_' + TB_NI_file + '_' + str(k) + '.out' + background) 
    fMGOPT.write(preamble +  NI_MGOpt_run_string + ' --seed ' + str(k*11) + '  >  temp_' + TB_NI_MGOpt_file + '_' + str(k) + '.out' + background)
    fMGOPT_LP.write(preamble_parallel + NI_MGOpt_LP_run_string + ' --seed ' + str(k*11) + '  >  temp_' + TB_NI_MGOpt_LP_file + '_' + str(k) + '.out' + background)
    fMGOPT_LP_LR.write(preamble_parallel + NI_MGOpt_LP_LR_run_string + ' --seed ' + str(k*11) + '  >  temp_' + TB_NI_MGOpt_LP_LR_file + '_' + str(k) + '.out' + background)

  ##
  # Dump footer, only for running background processes
  if ntests > 1:
    fAdam.write(Footer)
    fNI.write(Footer)
    fMGOPT.write(Footer)
    fMGOPT_LP.write(Footer)
    fMGOPT_LP_LR.write(Footer)

  ##
  # Dump cat command to collate output files 
  fAdam.write('\n\ncat  temp_' + TB_Adam_file + '*.out  >>  ' + TB_Adam_file + '.txt\n')
  fNI.write('\n\ncat  temp_' + TB_NI_file + '*.out  >>  ' + TB_NI_file + '.txt\n')
  fMGOPT.write('\n\ncat  temp_' + TB_NI_MGOpt_file + '*.out  >>  ' + TB_NI_MGOpt_file + '.txt\n')
  fMGOPT_LP.write('\n\ncat  temp_' + TB_NI_MGOpt_LP_file + '*.out  >>  ' + TB_NI_MGOpt_LP_file + '.txt\n')
  fMGOPT_LP_LR.write('\n\ncat  temp_' + TB_NI_MGOpt_LP_LR_file + '*.out  >>  ' + TB_NI_MGOpt_LP_LR_file + '.txt\n')

  ##
  # Dump runstrings to final .txt output
  fAdam.write('echo \" \" >> ' + TB_Adam_file + '.txt\n'  +  'echo Run String Used here is: >> ' + TB_Adam_file + '.txt\n'  +  'echo ' + Adam_run_string + ' >> ' + TB_Adam_file + '.txt')
  fNI.write('echo \" \" >> ' + TB_NI_file + '.txt \n'  +  'echo Run String Used here is: >> ' + TB_NI_file + '.txt\n'  +  'echo ' + NI_run_string + ' >> ' + TB_NI_file + '.txt')
  fMGOPT.write('echo \" \" >> ' + TB_NI_MGOpt_file + '.txt \n'  +  'echo Run String Used here is: >> ' + TB_NI_MGOpt_file + '.txt\n'  +  'echo ' + NI_MGOpt_run_string + ' >> ' + TB_NI_MGOpt_file + '.txt')
  fMGOPT_LP.write('echo \" \" >> ' + TB_NI_MGOpt_LP_file + '.txt \n'  +  'echo Run String Used here is: >> ' + TB_NI_MGOpt_LP_file + '.txt\n'  +  'echo ' + NI_MGOpt_LP_run_string + ' >> ' + TB_NI_MGOpt_LP_file + '.txt')
  fMGOPT_LP_LR.write('echo \" \" >> ' + TB_NI_MGOpt_LP_LR_file + '.txt \n'  +  'echo Run String Used here is: >> ' + TB_NI_MGOpt_LP_LR_file + '.txt\n'  +  'echo ' + NI_MGOpt_LP_LR_run_string + ' >> ' + TB_NI_MGOpt_LP_LR_file + '.txt')

  ##
  # Finish up
  fAdam.close()
  fNI.close()
  fMGOPT.close()
  fMGOPT_LP.close()
  fMGOPT_LP_LR.close()

  

if __name__ == '__main__':
  main()

