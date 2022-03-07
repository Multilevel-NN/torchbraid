import numpy as np
import os
import sys

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
  ntests = 12           # number of individual training runs to do
  nnodes = int(ntests)  # can run only one python script per node, unfortunately
  queue = 'pdebug'
  time = "00:05:00"

  ##
  # Run string parameters
  steps = 16
  channels = 8
  #
  sampratio = 1.0
  preserve_optim = 1
  #
  Adamepochs = 167
  NIepochs = 92
  MGOptV11epochs = 40
  MGOptV01epochs = 53

  # One-level NI, plain Adam
  Adam_run_string = ' main_mgopt.py --steps ' + str(steps) + ' --channels ' + str(channels) + ' --samp-ratio ' + str(sampratio) + ' --zero-init-guess 0 --mgopt-printlevel 1 --ni-levels 1 --lp-fwd-levels 1 --lp-bwd-levels 1 --mgopt-iter 0 --NIepochs ' + str(Adamepochs)
  TB_Adam_file = 'TB_onelevel_Adam'
  
  # Two-level NI 
  NI_run_string = ' main_mgopt.py --steps ' + str(steps) + ' --channels ' + str(channels) + ' --samp-ratio ' + str(sampratio) + ' --zero-init-guess 0 --mgopt-printlevel 1 --ni-levels 2 --lp-fwd-levels 1 --lp-bwd-levels 1 --mgopt-iter 0 --NIepochs ' + str(NIepochs)
  TB_NI_file = 'TB_NI'

  # Two-level, V(1,1)
  NI_MGOpt_V11_run_string = 'main_mgopt.py --steps ' + str(steps) + ' --channels ' + str(channels) + ' --samp-ratio ' + str(sampratio) + ' --zero-init-guess 0 --mgopt-printlevel 1 --ni-levels 2 --mgopt-levels 2 --mgopt-nrelax-pre 1 --mgopt-nrelax-post 1 --mgopt-nrelax-coarse 1 --lp-fwd-levels 1 --lp-bwd-levels 1 --lp-iters 1 --preserve-optim ' + str(preserve_optim) + ' --NIepochs 2 --epochs ' + str(MGOptV11epochs)
  TB_NI_MGOpt_V11_file = 'TB_NI_MGOpt_twolevel_V11'

  # Two-level, V(0,1)
  NI_MGOpt_V01_run_string = 'main_mgopt.py --steps ' + str(steps) + ' --channels ' + str(channels) + ' --samp-ratio ' + str(sampratio) + ' --zero-init-guess 0 --mgopt-printlevel 1 --ni-levels 2 --mgopt-levels 2 --mgopt-nrelax-pre 0 --mgopt-nrelax-post 1 --mgopt-nrelax-coarse 1 --lp-fwd-levels 1 --lp-bwd-levels 1 --lp-iters 1 --preserve-optim ' + str(preserve_optim) + ' --NIepochs 2 --epochs ' + str(MGOptV01epochs)
  TB_NI_MGOpt_V01_file = 'TB_NI_MGOpt_twolevel_V01'

  # Three-level, old line
  #NI_MGOpt_run_string = 'main_mgopt.py --steps ' + str(steps) + ' --channels ' + str(channels) + ' --samp-ratio ' + str(sampratio) + ' --zero-init-guess 0 --mgopt-printlevel 1 --ni-levels 3 --mgopt-levels 3 --mgopt-nrelax-pre 1 --mgopt-nrelax-post 1 --mgopt-nrelax-coarse 1 --lp-fwd-levels 1 --lp-bwd-levels 1 --lp-iters 1 --preserve-optim 1 --epochs 42 --NIepochs 2 '
  
  # Two-level LR, Keeping cost here equal is a bit tricky because of the LR
  NI_MGOpt_LR_run_string = 'main_mgopt.py --steps ' + str(steps) + ' --channels ' + str(channels) + ' --samp-ratio ' + str(sampratio) + ' --zero-init-guess 0 --mgopt-printlevel 1 --ni-levels 3 --mgopt-levels 3 --mgopt-nrelax-pre 1 --mgopt-nrelax-post 1 --mgopt-nrelax-coarse 1 --lp-fwd-cfactor 2 --lp-bwd-cfactor 2 --lp-fwd-levels 1 --lp-bwd-levels 1 --lp-bwd-finefcf --lp-bwd-relaxonlycg --lp-iters 1 --preserve-optim ' + str(preserve_optim) + ' --NIepochs 5 --epochs ' + str(MGOptV01epochs)
  TB_NI_MGOpt_LR_file = 'TB_NI_MGOpt_twolevel_LR'

  
  ##
  # Construct SBatch header
  Header = "#!/bin/bash\n" +\
           "#SBATCH -p " + queue + "\n" +\
           "#SBATCH -N "+str(nnodes) + "\n" +\
           "#SBATCH -c 36\n" +\
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
           "export PYTHONPATH=\"$HOME/2022_torchbraid/torchbraid/torchbraid\" \n" +\
           "export PYTHONPATH=\"$HOME/2022_torchbraid/torchbraid:$PYTHONPATH\" \n\n"

  Footer = "\nwait \n\n"

  ##
  # Open files, and dump header
  fAdam      = open(TB_Adam_file + '.sbatch', "w")
  fNI        = open(TB_NI_file + '.sbatch', "w")
  fMGOPT_V11 = open(TB_NI_MGOpt_V11_file + '.sbatch', "w")
  fMGOPT_V01 = open(TB_NI_MGOpt_V01_file + '.sbatch', "w")
  fMGOPT_LR  = open(TB_NI_MGOpt_LR_file + '.sbatch', "w")
  #
  fAdam.write(Header)
  fNI.write(Header)
  fMGOPT_V11.write(Header)
  fMGOPT_V01.write(Header)
  fMGOPT_LR.write(Header)

  ##
  # Dump our tests to each file
  preamble = 'srun -N 1 -n 1 python3 '
  for k in range(ntests):
    fAdam.write(preamble + Adam_run_string + ' --seed ' + str(k*11) + '  >  temp_' + TB_Adam_file + '_' + str(k) + '.out' + ' &\n')
    fNI.write(preamble + NI_run_string + ' --seed ' + str(k*11) + '  >  temp_' + TB_NI_file + '_' + str(k) + '.out' + ' &\n')
    fMGOPT_V11.write(preamble +  NI_MGOpt_V11_run_string + ' --seed ' + str(k*11) + '  >  temp_' + TB_NI_MGOpt_V11_file + '_' + str(k) + '.out' + ' &\n')
    fMGOPT_V01.write(preamble + NI_MGOpt_V01_run_string + ' --seed ' + str(k*11) + '  >  temp_' + TB_NI_MGOpt_V01_file + '_' + str(k) + '.out' + ' &\n')
    fMGOPT_LR.write(preamble + NI_MGOpt_LR_run_string + ' --seed ' + str(k*11) + '  >  temp_' + TB_NI_MGOpt_LR_file + '_' + str(k) + '.out' + ' &\n')

  ##
  # Dump footer
  fAdam.write(Footer)
  fNI.write(Footer)
  fMGOPT_V11.write(Footer)
  fMGOPT_V01.write(Footer)
  fMGOPT_LR.write(Footer)

  ##
  # Dump cat command to collate output files 
  fAdam.write('cat  temp_' + TB_Adam_file + '*.out  >>  ' + TB_Adam_file + '.txt\n')
  fNI.write('cat  temp_' + TB_NI_file + '*.out  >>  ' + TB_NI_file + '.txt\n')
  fMGOPT_V11.write('cat  temp_' + TB_NI_MGOpt_V11_file + '*.out  >>  ' + TB_NI_MGOpt_V11_file + '.txt\n')
  fMGOPT_V01.write('cat  temp_' + TB_NI_MGOpt_V01_file + '*.out  >>  ' + TB_NI_MGOpt_V01_file + '.txt\n')
  fMGOPT_LR.write('cat  temp_' + TB_NI_MGOpt_LR_file + '*.out  >>  ' + TB_NI_MGOpt_LR_file + '.txt\n')

  ##
  # Dump runstrings to final .txt output
  fAdam.write('echo \" \" >> ' + TB_Adam_file + '.txt\n'  +  'echo Run String Used here is: >> ' + TB_Adam_file + '.txt\n'  +  'echo ' + Adam_run_string + ' >> ' + TB_Adam_file + '.txt')
  fNI.write('echo \" \" >> ' + TB_NI_file + '.txt \n'  +  'echo Run String Used here is: >> ' + TB_NI_file + '.txt\n'  +  'echo ' + NI_run_string + ' >> ' + TB_NI_file + '.txt')
  fMGOPT_V11.write('echo \" \" >> ' + TB_NI_MGOpt_V11_file + '.txt \n'  +  'echo Run String Used here is: >> ' + TB_NI_MGOpt_V11_file + '.txt\n'  +  'echo ' + NI_MGOpt_V11_run_string + ' >> ' + TB_NI_MGOpt_V11_file + '.txt')
  fMGOPT_V01.write('echo \" \" >> ' + TB_NI_MGOpt_V01_file + '.txt \n'  +  'echo Run String Used here is: >> ' + TB_NI_MGOpt_V01_file + '.txt\n'  +  'echo ' + NI_MGOpt_V01_run_string + ' >> ' + TB_NI_MGOpt_V01_file + '.txt')
  fMGOPT_LR.write('echo \" \" >> ' + TB_NI_MGOpt_LR_file + '.txt \n'  +  'echo Run String Used here is: >> ' + TB_NI_MGOpt_LR_file + '.txt\n'  +  'echo ' + NI_MGOpt_LR_run_string + ' >> ' + TB_NI_MGOpt_LR_file + '.txt')

  ##
  # Finish up
  fAdam.close()
  fNI.close()
  fMGOPT_V11.close()
  fMGOPT_V01.close()
  fMGOPT_LR.close()

  

if __name__ == '__main__':
  main()

