from numpy import *
from matplotlib import pyplot as plt


def grab_losses_acc_and_NI_MGOPT_transitions(filename):
    f = open(filename)
    lines = f.readlines()
    f.close()
    
    glob_num_correct = []
    num_correct = []
    glob_accur = []
    accur = []
    glob_losses = []
    losses = []
    glob_train_losses = []
    train_losses = []
    for line in lines:
        
        # Does this line signify the end of an epoch, containing a Test loss number? (Validation set)
        if line.startswith('  Test set:'):
            loss_start = line.find(":", line.find(":")+1) + 2
            loss_end = line.find(",", loss_start)
            loss = float(line[loss_start:loss_end])
            losses.append(loss)
            
            acc_start = line.find("(", 0) + 1
            acc_end = line.find(")", acc_start) -1
            acc = float(line[acc_start:acc_end])
            accur.append(acc)

            num_corr_start = line.find("Accuracy: ", 0) + 10
            num_corr_end = line.find("/", num_corr_start)
            num_corr = float(line[num_corr_start:num_corr_end])
            num_correct.append(num_corr)
        
        # Does this line contain a Training data set loss number? 
        if line.find('Train Epoch:') > -1:
            loss_start = line.find("Loss: ", 0) + 6
            loss_end = line.find("Time Per Batch", loss_start)-1
            loss = float(line[loss_start:loss_end])
            train_losses.append(loss)


        ##
        # If you want to break up your plots per NI level and MGOpt comment in the below commented out lines
        #
        if line.startswith('Nested iteration steps:  '):
        #if line.startswith('Nested Iter Level:  '):
            if len(losses) > 0:
                glob_losses.append(losses)
                glob_train_losses.append(train_losses)
                glob_accur.append(accur)
                glob_num_correct.append(num_correct)
            losses = []
            train_losses = []
            accur = []
            num_correct = []
       #
       #if line.startswith('MG/Opt Solver'):
       #    if len(losses) > 0:
       #        glob_losses.append(losses)
       #        glob_train_losses.append(train_losses)
       #        glob_accur.append(accur)
       #        glob_num_correct.append(num_correct)
       #    losses = []
       #    train_losses = []
       #    accur = []
       #    num_correct = []
        
    ## 
    # Append any last data
    glob_losses.append(losses)
    glob_train_losses.append(train_losses)
    glob_accur.append(accur)
    glob_num_correct.append(num_correct)

    return glob_losses, glob_train_losses, glob_accur, glob_num_correct
##


## 
# Plot losses and number of missed data points for just the first dataset present in each file


colors = ['-k', '-m', '-b', '-r', '-c', '-g', '-y', '-k', '-m', '-b', '-r', '-c', '-g', '-y', '-k', '-m', '-b', '-r', '-c', '-g', '-y']
colors2 = ['--k', '--m', '--b', '--r', '--c', '--g', '--y', '--k', '--m', '--b', '--r', '--c', '--g', '--y', '--k', '--m', '--b', '--r', '--c', '--g', '--y']

glob_losses1, glob_train_losses1, glob_accur1, glob_num_correct1 = grab_losses_acc_and_NI_MGOPT_transitions('TB_NI.out')
glob_losses2, glob_train_losses2, glob_accur2, glob_num_correct2 = grab_losses_acc_and_NI_MGOPT_transitions('TB_NI_MGOpt.out')
glob_losses3, glob_train_losses3, glob_accur3, glob_num_correct3 = grab_losses_acc_and_NI_MGOPT_transitions('TB_NI_MGOpt_LR.out')

# Global constant representing how to scale the NI to be equiv. to MGOpt
scale_factor = float(len(glob_losses2[0])) / float(len(glob_losses1[0]))
total_val_examples = 2000

# Plot losses (training and validation
fig, ax1 = plt.subplots()
color_counter = 0
for k in range(1):#len(glob_losses1)):
    data1 = array(glob_losses1[k])
    data111 = array(glob_train_losses1[k])
    indys1 = arange( data1.shape[0] )
    indys111 = arange( data111.shape[0] )
    local_scale_factor = float(len(glob_losses1[0])) / float(len(glob_train_losses1[0]))
    data1[data1 == 0.0] = 1e-10
    ax1.semilogy(scale_factor*indys1, data1, colors[color_counter], label='NI')
    ax1.semilogy(scale_factor*local_scale_factor*indys111, data111, colors[color_counter], alpha=0.33)
    color_counter += 1
    #
    data2 = array(glob_losses2[k])
    data222 = array(glob_train_losses2[k])
    indys2 = arange(data2.shape[0])
    indys222 = arange(data222.shape[0])
    local_scale_factor = float(len(glob_losses2[0])) / float(len(glob_train_losses2[0]))
    data2[data2 == 0.0] = 1e-15
    ax1.semilogy(indys2, data2, colors[color_counter], label='NI+MG/Opt')
    ax1.semilogy(local_scale_factor*indys222, data222, colors[color_counter], alpha=0.33)
    color_counter += 1
    #
    data3 = array(glob_losses3[k])
    data333 = array(glob_train_losses3[k])
    indys3 = arange(data3.shape[0])
    indys333 = arange(data333.shape[0])
    local_scale_factor = float(len(glob_losses3[0])) / float(len(glob_train_losses3[0]))
    data3[data3 == 0.0] = 1e-15
    ax1.semilogy(indys3, data3, colors[color_counter], label='NI+MG/Opt+LocalRelax')
    ax1.semilogy(local_scale_factor*indys333, data333, colors[color_counter], alpha=0.33)
    color_counter += 1

ax1.set_xlabel('Work Units (3 Relaxations)', fontsize='large')
ax1.set_ylabel('Loss\n solid: validation,  shaded: training loss range', fontsize='large')
ax1.legend()
plt.savefig('compare_losses.png', pad_inches=0.12, bbox_inches='tight')


# Now plot number of missed validation test cases
fig3, ax3 = plt.subplots()
color_counter = 0
for k in range(1):#len(glob_losses1)):
    data11 = array(glob_num_correct1[k])
    indys1 = arange( data11.shape[0] )
    ax3.plot(scale_factor*indys1, 2000-data11, colors2[color_counter], label='NI')
    color_counter += 1
    #
    data22 = array(glob_num_correct2[k])
    indys2 = arange(data2.shape[0])
    ax3.plot(indys2, 2000-data22, colors2[color_counter], label='NI+MG/Opt')
    color_counter += 1
    #
    data33 = array(glob_num_correct3[k])
    indys3 = arange(data3.shape[0])
    ax3.plot(indys3, 2000-data33, colors2[color_counter], label='NI+MG/Opt+LocalRelax')
    color_counter += 1

ax3.set_xlabel('Work Units (3 Relaxations)', fontsize='large')
ax3.set_ylabel('Accuracy\nNumber Incorrect Validation Examples (2000 total)', fontsize='large')
ax3.legend()
plt.savefig('compare_accuracy.png', pad_inches=0.12, bbox_inches='tight')





## # Plot validation losses and num validation cases missed on left and right y-axes of a figure
## scale_factor = float(len(glob_losses2[0])) / float(len(glob_losses1[0]))
## total_val_examples = 2000
## 
## fig, ax1 = plt.subplots()
## color1 = 'tab:red'
## ax2 = ax1.twinx()
## color2 = 'tab:green'
## color_counter = 0
## for k in range(1):#len(glob_losses1)):
##     data1 = array(glob_losses1[k])
##     data11 = array(glob_num_correct1[k])
##     indys = arange( data1.shape[0] )
##     data1[data1 == 0.0] = 1e-10
##     ax1.semilogy(scale_factor*indys, data1, colors[color_counter])
##     ax2.plot(scale_factor*indys, 2000-data11, colors2[color_counter])
## 
##     color_counter += 1
##     #
##     data2 = array(glob_losses2[k])
##     data22 = array(glob_num_correct2[k])
##     indys = arange(data2.shape[0])
##     data2[data2 == 0.0] = 1e-10
##     ax1.semilogy(indys, data2, colors[color_counter])
##     ax2.plot(indys, 2000-data22, colors2[color_counter])
##     color_counter += 1
##     #
##     data3 = array(glob_losses3[k])
##     data33 = array(glob_num_correct3[k])
##     indys = arange(data3.shape[0])
##     data3[data3 == 0.0] = 1e-10
##     ax1.semilogy(indys, data3, colors[color_counter])
##     ax2.plot(indys, 2000-data33, colors2[color_counter])
##     color_counter += 1
## 
## 
## ax1.set_xlabel('Work Units (3 Relaxations)', fontsize='large')
## ax1.set_ylabel('Loss (solid lines)', fontsize='large', color = color1)
## ax1.tick_params(axis ='y', labelcolor = color1)
## ax2.tick_params(axis ='y', labelcolor = color2)
## ax2.set_ylabel('Accuracy, Num Mis-Classified (dotted lines) ', fontsize='large', color = color2)
## ax1.legend(['NI', 'NI+MG/Opt', 'NI+MG/Opt+LocalRelax'], fontsize='large')
## 
## plt.savefig('compare_global_losses.png', pad_inches=0.12, bbox_inches='tight')




##
# Old Plot showing how to plot each NI and MG/Opt level (have to uncomment stuff in helper fcn)
## plt.figure(1)
## color_counter = 0
## counter1 = 0
## counter2 = 0
## counter3 = 0
## for k in range(len(glob_accur1)):
##     data1 = array(glob_accur1[k])
##     indys = arange(counter1, counter1+data1.shape[0])
##     counter1 = counter1 + data1.shape[0]
##     plt.plot(scale_factor*indys, data1, colors[color_counter])
##     color_counter += 1
##     #
##     data2 = array(glob_accur2[k])
##     indys = arange(counter2, counter2+data2.shape[0])
##     counter2 = counter2 + data2.shape[0]
##     plt.plot(indys, data2, colors[color_counter])
##     color_counter += 1
##     #
##     data3 = array(glob_accur3[k])
##     indys = arange(counter3, counter3+data3.shape[0])
##     counter3 = counter3 + data3.shape[0]
##     plt.plot(indys, data3, colors[color_counter])
##     color_counter += 1

## plt.savefig('compare_global_accur.png', pad_inches=0.0, bbox_inches='tight')
##

