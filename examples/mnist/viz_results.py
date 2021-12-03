from numpy import *
from matplotlib import pyplot as plt

def convert_data_for_plotting( losses, accur, num_correct, train_losses):
    '''
    Take the list-of-lists losses, accur, num_correct, and train_losses

    Convert to numpy arrays for potting, and
    return x-axis data for plotting together
    '''
    ndata_sets = len(losses)
    nvalidation_points = len(losses[0])
    ntraining_points = len(train_losses[0])

    losses = hstack(losses).reshape(ndata_sets, nvalidation_points)
    accur = hstack(accur).reshape(ndata_sets, nvalidation_points)
    num_correct = hstack(num_correct).reshape(ndata_sets, nvalidation_points)
    train_losses = hstack(train_losses).reshape(ndata_sets, ntraining_points)

    # Compute standard deviation
    #std_losses = std(losses, axis=0)
    #std_accur = std(accur, axis=0)
    #std_num_correct = std(num_correct, axis=0)
    #std_train_losses = std(train_losses, axis=0)

    validation_x_axis = arange( nvalidation_points )
    training_x_axis = arange( ntraining_points ) * (float(nvalidation_points-1)/float(ntraining_points-1))

    return losses, accur, num_correct, train_losses, validation_x_axis, training_x_axis


def plot_losses(ax, losses, train_losses, validation_x_axis, training_x_axis, scale_factor, color, label=''):
    '''
    Plot training and validation losses
    '''

    losses[losses == 0.0] = 1e-10
    data = mean(losses, axis=0)
    ax.semilogy(scale_factor*validation_x_axis, data, '-'+color, label=label+' validation mean')
    data_up = mean(losses, axis=0) + std_dev_scale*std(losses, axis=0)
    data_down = mean(losses, axis=0) - std_dev_scale*std(losses, axis=0)
    ax.fill_between(scale_factor*validation_x_axis, data_up, data_down, color=color, alpha=0.33)
    #
    train_losses[train_losses == 0.0] = 1e-10
    data = mean(train_losses, axis=0)
    ax.semilogy(scale_factor*training_x_axis, data, '-'+color, label=label+' train mean', alpha=0.33)
    # Skip plotting the bounds around losses
    #data_up = mean(train_losses, axis=0) + std_dev_scale*std(train_losses, axis=0)
    #data_down = mean(train_losses, axis=0) - std_dev_scale*std(train_losses, axis=0)
    #ax.fill_between(training_x_axis, data_up, data_down, color=color, alpha=0.33)


def plot_validation_misclassified(ax, num_correct, validation_x_axis, scale_factor, color, label='NI'):
    '''
    Plot number misclassified in validation set 
    '''

    data = mean(num_correct, axis=0)
    ax.plot(scale_factor*validation_x_axis, data, '-'+color, label=label+' validation mean')
    data_up = mean(num_correct, axis=0) + std_dev_scale*std(num_correct, axis=0)
    data_down = mean(num_correct, axis=0) - std_dev_scale*std(num_correct, axis=0)
    ax.fill_between(scale_factor*validation_x_axis, data_up, data_down, color=color, alpha=0.33)


def grab_losses_acc_and_NI_MGOPT_transitions(filename):
    '''
    Return validation set losses     : glob_losses
           training set losses       : glob_train_losses
           validation set accuracies : glob_accur
           validation set number     : glob_num_correct
              correctly classified

    '''
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



colors = ['k', 'm', 'b', 'r', 'c', 'g', 'y', 'k', 'm', 'b', 'r', 'c', 'g', 'y', 'k', 'm', 'b', 'r', 'c', 'g', 'y']
colors2 = ['-k', '-m', '-b', '-r', '-c', '-g', '-y', '-k', '-m', '-b', '-r', '-c', '-g', '-y', '-k', '-m', '-b', '-r', '-c', '-g', '-y']
colors3 = ['--k', '--m', '--b', '--r', '--c', '--g', '--y', '--k', '--m', '--b', '--r', '--c', '--g', '--y', '--k', '--m', '--b', '--r', '--c', '--g', '--y']

# test_results/simple_ls_67db6bbf/  test result for hash 67db6bbf using the defaults in that repo, nothing else, simple line-search
losses1, train_losses1, accur1, num_correct1 = grab_losses_acc_and_NI_MGOPT_transitions('test_results/simple_ls_67db6bbf/TB_NI.out')
losses2, train_losses2, accur2, num_correct2 = grab_losses_acc_and_NI_MGOPT_transitions('test_results/simple_ls_67db6bbf/TB_NI_MGOpt.out')
losses3, train_losses3, accur3, num_correct3 = grab_losses_acc_and_NI_MGOPT_transitions('test_results/simple_ls_67db6bbf/TB_NI_MGOpt_LR.out')

# Convert to nice for plotting arrays
losses1, accur1, num_correct1, train_losses1, validation_x_axis1, training_x_axis1 = \
        convert_data_for_plotting(losses1, accur1, num_correct1, train_losses1)
losses2, accur2, num_correct2, train_losses2, validation_x_axis2, training_x_axis2 = \
    convert_data_for_plotting(losses2, accur2, num_correct2, train_losses2)
losses3, accur3, num_correct3, train_losses3, validation_x_axis3, training_x_axis3 = \
    convert_data_for_plotting(losses3, accur3, num_correct3, train_losses3)


# Total number of validation examples
total_val_examples = 2000
# Plot shaded regions encompassing 90%
std_dev_scale = 2.0#1.645 
# Find scaling factor between NI and NI*MGOpt
scale_factor = float(validation_x_axis2.shape[0])/float(validation_x_axis1.shape[0])

# Plot Losses
fig1, ax1 = plt.subplots(1,1)
# Plot NI losses, training and validation
plot_losses(ax1, losses1, train_losses1, validation_x_axis1, training_x_axis1, scale_factor, colors[0], label='NI')
# Plot NI+MGOpt losses, training and validation
plot_losses(ax1, losses2, train_losses2, validation_x_axis2, training_x_axis2, 1.0, colors[1], label='NI+MGOpt')
# Labels and such
ax1.set_xlabel('Work Units (4 Relaxations)', fontsize='large')
ax1.set_ylabel('Loss', fontsize='large')
ax1.legend(loc='lower left')
plt.savefig('compare_losses.png', pad_inches=0.12, bbox_inches='tight')
#

# Plot number of missed validation test cases
fig2, ax2 = plt.subplots(1,1)
plot_validation_misclassified(ax2, total_val_examples-num_correct1, validation_x_axis1, scale_factor, colors[0], label='NI')
plot_validation_misclassified(ax2, total_val_examples-num_correct2, validation_x_axis2, 1.0, colors[1], label='NI+MGOpt')
#plot_validation_misclassified(ax2, total_val_examples-num_correct3, validation_x_axis3, scale_factor, colors[0], label='NI+MGOpt+LocalRelax')
# Labels and such
ax2.set_xlabel('Work Units (4 Relaxations)', fontsize='large')
ax2.set_ylabel('Accuracy\nNumber Incorrect Validation Examples (2000 total)', fontsize='large')
ax2.legend(loc='lower left')
plt.savefig('compare_accuracy.png', pad_inches=0.12, bbox_inches='tight')


#  DOUBLE CHECK PLOTS!!!
# Add some plots to PowerPoint
# - NI vs. MG/Opt
# - MG/Opt with three different CGC
#
# Git commit

