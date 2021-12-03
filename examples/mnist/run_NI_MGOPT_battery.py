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
  $ python3  run_NI_MGOPT_battery.py  run_NI >  temp.out  2>&1  & 

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
  #      This translates to a cost in terms of fine-grid optimizations of 
  #        1/2*(5 + total_nrelax*12) + (5 + total_nrelax*12)
  #        -- 1/2: size of coarse-grid
  #        -- (5 + total_nrelax*12):  refers to the 5 NI steps (NI does only one relax each training sweep)
  #                                   and to the 12 MG/Opt epochs with total_nrelax sweeps 
  #        -- This yields   1./2.*(5 + 4*12) + (5 + 4*12)  =  79.5
  #           Or, we do (5 + 4*12) = 53 epochs on each NI level so that all solvers do a similar amount of "training"
  NI_run_string = ' main_mgopt.py --steps 16 --samp-ratio 0.2 --mgopt-printlevel 1 --ni-levels 2 --lp-fwd-levels 1 --lp-bwd-levels 1 --mgopt-iter 0 --NIepochs 53'
  NI_MGOpt_run_string = 'main_mgopt.py --steps 16 --samp-ratio 0.2 --mgopt-printlevel 1 --ni-levels 2 --mgopt-levels 2 --mgopt-nrelax-pre 2 --mgopt-nrelax-post 2 --lp-fwd-levels 1 --lp-bwd-levels 1 --lp-iters 1  --epochs 12 --NIepochs 5 '
  NI_MGOpt_LR_run_string = 'main_mgopt.py --steps 16 --samp-ratio 0.2 --mgopt-printlevel 1 --ni-levels 2 --mgopt-levels 2 --mgopt-nrelax-pre 2 --mgopt-nrelax-post 2 --lp-fwd-cfactor 2 --lp-bwd-cfactor 2 --lp-fwd-levels 1 --lp-bwd-levels 1 --lp-bwd-finefcf --lp-bwd-relaxonlycg --lp-iters 1  --epochs 12 --NIepochs 5'

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

