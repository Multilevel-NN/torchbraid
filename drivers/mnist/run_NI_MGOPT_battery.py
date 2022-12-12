import numpy as np
import os
import sys

def main():
  '''
  This function runs multiple tests for NI, NI+MGOPT, and NI+MGOPT+LocalRelax,
  depending on the chosen user options.

  Run the NI experiments ntests times
  $ python3  run_NI_MGOPT_battery.py  run_NI

  Run the NI+MGOPT experiments ntests times
  $ python3  run_NI_MGOPT_battery.py  run_NI_MGOPT

  Run the NI+MGOPT+LocalRelax experiments ntests times
  $ python3  run_NI_MGOPT_battery.py  run_NI_MGOPT_LR

  Cat any temporary files into the global output file, and remove temporary files
  $ python3  run_NI_MGOPT_battery.py  collate_output

  If you want to leave it run overnight, redirect std error and out to a file with
  $ python3  run_NI_MGOPT_battery.py  run_NI_MGOpt >  temp.out  2>&1  & 

  '''
  
  ##
  # Use of this script is a bit clunky, you have to comment in the correct
  # run_string that you want to use below.  We record the description of the
  # run-strings here
  
  # Test 1: Basic two-level experiment on MNIST
  #  - [8, 16] layers 
  #  - 2 NRelax (to avoid exactness if doing local relax)
  #  - Do less NI, say only 5 epochs, so that MG/Opt has work to do
  #  - Do 12 MG/Opt epochs after that
  #  - This translates to a cost in terms of fine-grid optimizations for straight Adam/NI of
  #    12 MG/Opt epochs  X  (4 fine-grid relaxations + 5 coarse-grid relaxations)  +   5 NI epochs X 2 Levels
  #    = 118 total relaxations
  #  - We eventually 154 relaxations in NI, though, as we lost track of costs and started doing more epochs for MGOpt 
  #    in variants than the original.  In particular, we do 
  #    -->  154 = 48*3 + 10 relaxations   (which corresponds to 48 epoch tests below)
  #    -->  But we do (154/2) = 77 epochs on each level
  #
  # Old string: NI_run_string = ' main_mgopt.py --steps 16 --samp-ratio 0.2 --mgopt-printlevel 1 --ni-levels 2 --lp-fwd-levels 1 --lp-bwd-levels 1 --mgopt-iter 0 --NIepochs 53'
  #             this was a fair comparison with the 12 epoch tests
  # New strings for two- and single level: 
  #NI_run_string = ' main_mgopt.py --steps 16 --samp-ratio 0.2 --mgopt-printlevel 1 --ni-levels 1 --lp-fwd-levels 1 --lp-bwd-levels 1 --mgopt-iter 0 --NIepochs 154'
  #NI_run_string = ' main_mgopt.py --steps 16 --samp-ratio 0.2 --mgopt-printlevel 1 --ni-levels 2 --lp-fwd-levels 1 --lp-bwd-levels 1 --mgopt-iter 0 --NIepochs 77'
  ####
  #NI_MGOpt_run_string = 'main_mgopt.py --steps 16 --samp-ratio 0.2 --mgopt-printlevel 1 --ni-levels 2 --mgopt-levels 2 --mgopt-nrelax-pre 2 --mgopt-nrelax-post 2 --lp-fwd-levels 1 --lp-bwd-levels 1 --lp-iters 1  --epochs 12 --NIepochs 5 '
  #
  # Variant of MGOpt where we do only 1 relaxation, but 2x epochs 
  #NI_MGOpt_run_string = 'main_mgopt.py --steps 16 --samp-ratio 0.2 --mgopt-printlevel 1 --ni-levels 2 --mgopt-levels 2 --mgopt-nrelax-pre 1 --mgopt-nrelax-post 1 --lp-fwd-levels 1 --lp-bwd-levels 1 --lp-iters 1  --epochs 24 --NIepochs 5 '
  #
  # Variant of MGOpt where we do only 1 relaxation and 1 coarse-grid relaxation, but 4x epochs 
  #NI_MGOpt_run_string = 'main_mgopt.py --steps 16 --samp-ratio 0.2 --mgopt-printlevel 1 --ni-levels 2 --mgopt-levels 2 --mgopt-nrelax-pre 1 --mgopt-nrelax-post 1 --mgopt-nrelax-coarse 1 --lp-fwd-levels 1 --lp-bwd-levels 1 --lp-iters 1  --epochs 48 --NIepochs 5 '
  #
  # Variant of MGOpt where we do three-levels only 1 relaxation and 1 coarse-grid relaxation, but 4x epochs 
  #NI_MGOpt_run_string = 'main_mgopt.py --steps 16 --samp-ratio 0.2 --mgopt-printlevel 1 --ni-levels 3 --mgopt-levels 3 --mgopt-nrelax-pre 1 --mgopt-nrelax-post 1 --mgopt-nrelax-coarse 1 --lp-fwd-levels 1 --lp-bwd-levels 1 --lp-iters 1  --epochs 48 --NIepochs 5 '
  #
  # Variant of MGOpt where we do four-levels only 1 relaxation and 1 coarse-grid relaxation, but 4x epochs 
  #NI_MGOpt_run_string = 'main_mgopt.py --steps 16 --samp-ratio 0.2 --mgopt-printlevel 1 --ni-levels 4 --mgopt-levels 4 --mgopt-nrelax-pre 1 --mgopt-nrelax-post 1 --mgopt-nrelax-coarse 1 --lp-fwd-levels 1 --lp-bwd-levels 1 --lp-iters 1  --epochs 48 --NIepochs 5 '
  ####
  #NI_MGOpt_LR_run_string = 'main_mgopt.py --steps 16 --samp-ratio 0.2 --mgopt-printlevel 1 --ni-levels 2 --mgopt-levels 2 --mgopt-nrelax-pre 2 --mgopt-nrelax-post 2 --lp-fwd-cfactor 2 --lp-bwd-cfactor 2 --lp-fwd-levels 1 --lp-bwd-levels 1 --lp-bwd-finefcf --lp-bwd-relaxonlycg --lp-iters 1  --epochs 24 --NIepochs 10'
  #
  # Variant of LR where we do only one relaxation, but more epochs and three levels (we want to fully solve the coarsest level)
  #NI_MGOpt_LR_run_string = 'main_mgopt.py --steps 16 --samp-ratio 0.2 --mgopt-printlevel 1 --ni-levels 3 --mgopt-levels 3 --mgopt-nrelax-pre 2 --mgopt-nrelax-post 2 --mgopt-nrelax-coarse 1 --lp-fwd-cfactor 2 --lp-bwd-cfactor 2 --lp-fwd-levels 1 --lp-bwd-levels 1 --lp-bwd-finefcf --lp-bwd-relaxonlycg --lp-iters 1  --epochs 48 --NIepochs 5'


  # Test 2: Basic two-level experiment on MNIST
  #   - Same as Test 1 EXCEPT
  #   - Only use all of MNIST  and  decrease NIepochs from 5 to 2 
  #     (so that NI still leaves space for MGOpt to improve), this results in one more epoch of MGOpt to keep the work about the same.
  #   - Do better job of keeping the cost the same
  #     -- Cost of 1 MGOPT epoch plus NI bootstrapping:
  #        (2 total fine-grid relaxations  + 1 g_h  + 1 coarse-grid relaxation + g_H  +  Line-search cost)  +   NI epochs X 2 Levels
  #        In terms of fine-grid gradient evaluations,         
  #        (2                              + 1      + 0.55                     + 0.55 +  Line-search cost)  +   NI epochs X 1.6
  #        (4.1 + Line-search cost) + NI epochs X 1.6  

  #     -- Cost for simple line-search, letting the cost be about 1 optimizer steps is
  #        166 = 32*(4.1 + 1.0) + 3.2
  #     -- Cost using no line search is equal when doing 40 MGOpt epochs and 2 NI epochs
  #        167 = 40*(4.1 + 0.0) + 3.2
  # 
  #     -- Do only 1 relaxation (post relaxation) and assume no significant line search cost, then overall cost is
  #        167 = 53*(3.1 + 0.0) + 3.2
  #
  #     -- NI epochs are then
  #        167 epochs for single level
  #        92 epochs for two levels (coarse grid is a little bit cheaper, so allow for a bit more epochs, this value computed based on num parameters) 
  #
  # Two-level NI 
  NI_run_string = ' main_mgopt.py --steps 16 --channels 8 --samp-ratio 1.0 --zero-init-guess 0 --mgopt-printlevel 1 --ni-levels 2 --lp-fwd-levels 1 --lp-bwd-levels 1 --mgopt-iter 0 --NIepochs 92'
  # One-level NI (plain Adam)
  #NI_run_string = ' main_mgopt.py --steps 16 --channels 8 --samp-ratio 1.0 --zero-init-guess 0 --mgopt-printlevel 1 --ni-levels 1 --lp-fwd-levels 1 --lp-bwd-levels 1 --mgopt-iter 0 --NIepochs 167'
  #######
  # Two-level
  NI_MGOpt_run_string = 'main_mgopt.py --steps 16 --channels 8 --samp-ratio 1.0 --zero-init-guess 0 --mgopt-printlevel 1 --ni-levels 2 --mgopt-levels 2 --mgopt-nrelax-pre 0 --mgopt-nrelax-post 1 --mgopt-nrelax-coarse 1 --lp-fwd-levels 1 --lp-bwd-levels 1 --lp-iters 1 --preserve-optim 1 --epochs 54 --NIepochs 0 '
  #
  # Three-level
  #NI_MGOpt_run_string = 'main_mgopt.py --steps 16 --channels 8 --samp-ratio 1.0 --zero-init-guess 0 --mgopt-printlevel 1 --ni-levels 3 --mgopt-levels 3 --mgopt-nrelax-pre 1 --mgopt-nrelax-post 1 --mgopt-nrelax-coarse 1 --lp-fwd-levels 1 --lp-bwd-levels 1 --lp-iters 1 --preserve-optim 1 --epochs 42 --NIepochs 2 '
  #######
  # Keeping cost here equal is a bit tricky because of the LR, but it's roughly the same
  NI_MGOpt_LR_run_string = 'main_mgopt.py --steps 16 --channels 8 --samp-ratio 1.0 --zero-init-guess 0 --mgopt-printlevel 1 --ni-levels 3 --mgopt-levels 3 --mgopt-nrelax-pre 1 --mgopt-nrelax-post 1 --mgopt-nrelax-coarse 1 --lp-fwd-cfactor 2 --lp-bwd-cfactor 2 --lp-fwd-levels 1 --lp-bwd-levels 1 --lp-bwd-finefcf --lp-bwd-relaxonlycg --lp-iters 1 --preserve-optim 1 --epochs 52 --NIepochs 5'
  
  # Test <>
  # - <>
  # - <>
  #NI_run_string = ' '
  #NI_MGOpt_run_string = ' '
  #NI_MGOpt_LR_run_string = ' '

  ######################################################################################
  ######################################################################################

  ##
  # Number of tests to run and file names 
  ntests = 12
  TB_NI_file = 'TB_NI'
  temp_TB_NI_file = 'temp_' + TB_NI_file 
  #
  TB_NI_MGOpt_file = 'TB_NI_MGOpt'
  temp_TB_NI_MGOpt_file = 'temp_' + TB_NI_MGOpt_file 
  #
  TB_NI_MGOpt_LR_file = 'TB_NI_MGOpt_LR'
  temp_TB_NI_MGOpt_LR_file = 'temp_' + TB_NI_MGOpt_LR_file 
  
  ##
  # Process command line arguments
  if sys.argv[1] == 'run_NI':
    run_string = NI_run_string
    temp_file = temp_TB_NI_file 
  
  elif sys.argv[1] == 'run_NI_MGOpt':
    run_string = NI_MGOpt_run_string
    temp_file = temp_TB_NI_MGOpt_file 
  
  elif sys.argv[1] == 'run_NI_MGOpt_LR':
    run_string = NI_MGOpt_LR_run_string
    temp_file = temp_TB_NI_MGOpt_LR_file

  elif sys.argv[1] == 'collate_output':
    # cat all possible temporary output files together, removing temp files
    #
    # This order of the three test cases is important, or else the cat-ing of files get's confused 

    # MGOpt+LR
    c = os.system('cat  ' + temp_TB_NI_MGOpt_LR_file + '*  >>  ' + TB_NI_MGOpt_LR_file  + '.out')
    if c == 0:
      os.system('rm  ' + temp_TB_NI_MGOpt_LR_file + '*')
      os.system('echo  >>  ' + TB_NI_MGOpt_LR_file  + '.out')
      os.system('echo Run String Used here is:  >>  ' + TB_NI_MGOpt_LR_file + '.out')
      os.system('echo ' + NI_MGOpt_LR_run_string + '  >>  ' + TB_NI_MGOpt_LR_file  + '.out')
  
    # MGOpt
    c = os.system('cat  ' + temp_TB_NI_MGOpt_file + '*  >>   ' + TB_NI_MGOpt_file + '.out')
    if c == 0:
      os.system('rm  ' + temp_TB_NI_MGOpt_file + '*')
      os.system('echo  >>  ' + TB_NI_MGOpt_file  + '.out')
      os.system('echo Run String Used here is:  >>  ' + TB_NI_MGOpt_file + '.out')
      os.system('echo ' + NI_MGOpt_run_string + '  >>  ' + TB_NI_MGOpt_file  + '.out')
    
    # NI
    c = os.system('cat  ' + temp_TB_NI_file + '*  >>  ' + TB_NI_file + '.out')
    if c == 0:  # if cat call returned no error
      os.system('rm  ' + temp_TB_NI_file + '*')
      os.system('echo  >>  ' + TB_NI_file  + '.out')
      os.system('echo Run String Used here is:  >>  ' + TB_NI_file + '.out')
      os.system('echo ' + NI_run_string + '  >>  ' + TB_NI_file  + '.out')
     
  else:
    print("INCORRECT USAGE -- ABORTING")

  # Debug output
  #print(run_string)
  #print(temp_file)

  ##
  # If running tests
  if sys.argv[1].startswith('run_NI'):
    for k in range(ntests):
      full_run_string = 'python3 ' + run_string + ' --seed ' + str(k*11) + '  >  ' +\
                         temp_file + '_' + str(k) + '.out' ## + ' &'  ## add the ampersand to run everything concurrently
      os.system(full_run_string)
      
      # Debug output
      print("\nSubmitting job " + str(k) + " :\n $ " + full_run_string, flush=True)


if __name__ == '__main__':
  main()

