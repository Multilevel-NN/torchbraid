from numpy import *
import numpy
from matplotlib import pyplot as plt

def convert_data_for_plotting( losses, accur, num_correct, train_losses):
    '''
    Take the list-of-lists losses, accur, num_correct, and train_losses

    Convert to numpy arrays for plotting
    '''
    ndata_sets = len(losses)
    nvalidation_points = len(losses[0])
    ntraining_points = len(train_losses[0])

    losses = hstack(losses).reshape(ndata_sets, nvalidation_points)
    accur = hstack(accur).reshape(ndata_sets, nvalidation_points)
    num_correct = hstack(num_correct).reshape(ndata_sets, nvalidation_points)
    train_losses = hstack(train_losses).reshape(ndata_sets, ntraining_points)

    return losses, accur, num_correct, train_losses


def plot_losses(ax, losses, train_losses, validation_x_axis, training_x_axis, color, label=''):
    '''
    Plot training and validation losses
    '''

    # 1.645 is for 90%...but this can lead to plotting artifacts.  Jut go with 1 std dev.
    
    # Uncomment this block to plot the validation losses
    #losses[losses == 0.0] = 1e-10
    #data = mean(losses, axis=0)
    #ax.semilogy(validation_x_axis, data, '-'+color, label=label+' validation mean')
    #data_up = amax(losses, axis=0)
    #data_down = amin(losses, axis=0)
    #ax.fill_between(validation_x_axis, data_up, data_down, color=color, alpha=0.33)
    #
    train_losses[train_losses == 0.0] = 1e-10
    data = mean(train_losses, axis=0)
    ax.semilogy(training_x_axis, data, '-'+color, label=label+' train mean', alpha=0.33)
    # Skip plotting the bounds around losses
    #data_up = amax(train_losses, axis=0)
    #data_down = amin(train_losses, axis=0)
    #ax.fill_between(training_x_axis, data_up, data_down, color=color, alpha=0.33)


def plot_validation_misclassified(ax, num_correct, validation_x_axis, color, label):
    '''
    Plot number misclassified in validation set 
    '''
    data = mean(num_correct, axis=0)
    ax.semilogy(validation_x_axis, data, '-'+color, label=label+' mean')
    
    # Using the std dev can yeild large regions where no data points exist, use max/min
    #data_up = mean(num_correct, axis=0) + 1.0*std(num_correct, axis=0)
    #data_down = mean(num_correct, axis=0) - 1.0*std(num_correct, axis=0)
    data_up = amax(num_correct, axis=0)
    data_down = amin(num_correct, axis=0)
    
    ax.fill_between(validation_x_axis, data_up, data_down, color=color, alpha=0.15)


def plot_accur(ax, accur, validation_x_axis, color, label):
    '''
    Plot the accuracy 
    '''
    data = mean(accur, axis=0)
    #data = numpy.max(accur, axis=0)
    ax.plot(validation_x_axis, data, '-'+color, label=label+' mean')
    
    # Using the std dev can yeild large regions where no data points exist, use max/min
    #data_up = mean(accur, axis=0) + 1.0*std(accur, axis=0)
    #data_down = mean(accur, axis=0) - 1.0*std(accur, axis=0)
    data_up = amax(accur, axis=0)
    data_down = amin(accur, axis=0)
    
    ax.fill_between(validation_x_axis, data_up, data_down, color=color, alpha=0.15)



def grab_losses_acc_and_NI_MGOpt_transitions(filename):
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

    MGOpt_start_val = -1
    MGOpt_start_train = -1
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
        # Record where MG/Opt starts
        if line.startswith('MG/Opt Solver'):
            if MGOpt_start_val == -1:
                MGOpt_start_val = len(losses)
                MGOpt_start_train = len(train_losses)
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

    return glob_losses, glob_train_losses, glob_accur, glob_num_correct, MGOpt_start_val, MGOpt_start_train
##



colors = ['k', 'm', 'b', 'c', 'r', 'g', 'y', 'k', 'm', 'b', 'c', 'r', 'g', 'y', 'k', 'm', 'b', 'r', 'c', 'g', 'y']
colors2 = ['-k', '-m', '-b', '-c', '-r', '-g', '-y', '-k', '-m', '-b', '-c', '-r', '-g', '-y', '-k', '-m', '-b', '-r', '-c', '-g', '-y']
colors3 = ['--k', '--m', '--b', '--c', '--r', '--g', '--y', '--k', '--m', '--b', '--c', '--r', '--g', '--y', '--k', '--m', '--b', '--r', '--c', '--g', '--y']

# Turn on the 20% MNIST plots
if False:

    # test_results/simple_ls_67db6bbf/  test result for hash 67db6bbf using the defaults in that repo, nothing else, simple line-search
    losses1, train_losses1, accur1, num_correct1, MGOpt_start_val1, MGOpt_start_train1 = \
            grab_losses_acc_and_NI_MGOpt_transitions('test_results/simple_ls_67db6bbf/TB_NI.out')
    losses2, train_losses2, accur2, num_correct2, MGOpt_start_val2, MGOpt_start_train2 = \
            grab_losses_acc_and_NI_MGOpt_transitions('test_results/simple_ls_67db6bbf/TB_NI_MGOpt.out')
    #losses3, train_losses3, accur3, num_correct3,  MGOopt_start_val3, MGOpt_start_train3 = \
    #        grab_losses_acc_and_NI_MGOpt_transitions('test_results/simple_ls_67db6bbf/TB_NI_MGOpt_LR.out')
    losses4, train_losses4, accur4, num_correct4, MGOpt_start_val4, MGOpt_start_train4 = \
           grab_losses_acc_and_NI_MGOpt_transitions('test_results/simple_ls_67db6bbf/TB_NI_MGOpt_damped_CGC_0_1.out')
    losses5, train_losses5, accur5, num_correct5, MGOpt_start_val5, MGOpt_start_train5 = \
           grab_losses_acc_and_NI_MGOpt_transitions('test_results/simple_ls_67db6bbf/TB_NI_MGOpt_damped_CGC_0_01.out')
    losses6, train_losses6, accur6, num_correct6, MGOpt_start_val6, MGOpt_start_train6 = \
           grab_losses_acc_and_NI_MGOpt_transitions('test_results/simple_ls_67db6bbf/TB_NI_MGOpt_damped_CGC_0_000003.out')
    losses7, train_losses7, accur7, num_correct7, MGOpt_start_val7, MGOpt_start_train7 = \
           grab_losses_acc_and_NI_MGOpt_transitions('test_results/simple_ls_67db6bbf/TB_NI_MGOpt_1_fine_grid_relax.out')
    losses8, train_losses8, accur8, num_correct8, MGOpt_start_val8, MGOpt_start_train8 = \
           grab_losses_acc_and_NI_MGOpt_transitions('test_results/simple_ls_67db6bbf/TB_NI_MGOpt_1_fine_and_coarse_grid_relax_damped_CGC_0_01.out')
    losses9, train_losses9, accur9, num_correct9, MGOpt_start_val9, MGOpt_start_train9 = \
           grab_losses_acc_and_NI_MGOpt_transitions('test_results/simple_ls_67db6bbf/TB_NI_MGOpt_1_fine_and_coarse_grid_relax.out')
    losses10, train_losses10, accur10, num_correct10, MGOpt_start_val10, MGOpt_start_train10 = \
           grab_losses_acc_and_NI_MGOpt_transitions('test_results/simple_ls_67db6bbf/TB_NI_single_level_Adam.out')
    losses11, train_losses11, accur11, num_correct11, MGOpt_start_val11, MGOpt_start_train11 = \
           grab_losses_acc_and_NI_MGOpt_transitions('test_results/simple_ls_67db6bbf/TB_NI_MGOpt_three_level.out')
    losses12, train_losses12, accur12, num_correct12, MGOpt_start_val12, MGOpt_start_train12 = \
           grab_losses_acc_and_NI_MGOpt_transitions('test_results/simple_ls_67db6bbf/TB_NI_MGOpt_fourlevel.out')

    # Convert to nice for plotting arrays
    losses1, accur1, num_correct1, train_losses1 = \
            convert_data_for_plotting(losses1, accur1, num_correct1, train_losses1)
    losses10, accur10, num_correct10, train_losses10 = \
            convert_data_for_plotting(losses10, accur10, num_correct10, train_losses10)
    losses2, accur2, num_correct2, train_losses2 = \
        convert_data_for_plotting(losses2, accur2, num_correct2, train_losses2)
    #losses3, accur3, num_correct3, train_losses3 = \
    #    convert_data_for_plotting(losses3, accur3, num_correct3, train_losses3)
    losses4, accur4, num_correct4, train_losses4 = \
        convert_data_for_plotting(losses4, accur4, num_correct4, train_losses4)
    losses5, accur5, num_correct5, train_losses5 = \
        convert_data_for_plotting(losses5, accur5, num_correct5, train_losses5)
    losses6, accur6, num_correct6, train_losses6 = \
        convert_data_for_plotting(losses6, accur6, num_correct6, train_losses6)
    losses7, accur7, num_correct7, train_losses7 = \
        convert_data_for_plotting(losses7, accur7, num_correct7, train_losses7)
    # Test set 7 does twice as many (but cheaper epochs)  So for plotting, we just take every other MGOpt epoch
    losses7 = losses7[:, [i for i in range(10)] + [i for i in range(10,34,2)] ]
    accur7 = accur7[:, [i for i in range(10)] + [i for i in range(10,34,2)] ]
    num_correct7 = num_correct7[:, [i for i in range(10)] + [i for i in range(10,34,2)] ]
    train_losses7 = train_losses7[:, [i for i in range(210)] + [i for i in range(210,714,2)] ]
    #
    losses8, accur8, num_correct8, train_losses8 = \
        convert_data_for_plotting(losses8, accur8, num_correct8, train_losses8)
    # Test set 8 does four as many (but cheaper epochs)  So for plotting, we just take every other MGOpt epoch
    losses8 = losses8[:, [i for i in range(10)] + [i for i in range(10,58,4)] ]
    accur8 = accur8[:, [i for i in range(10)] + [i for i in range(10,58,4)] ]
    num_correct8 = num_correct8[:, [i for i in range(10)] + [i for i in range(10,58,4)] ]
    train_losses8 = train_losses8[:, [i for i in range(210)] + [i for i in range(210,1218,4)] ]
    # Test set 9 does four as many (but cheaper epochs)  So for plotting, we just take every other MGOpt epoch
    losses9, accur9, num_correct9, train_losses9 = \
            convert_data_for_plotting(losses9, accur9, num_correct9, train_losses9)
    losses9 = losses9[:, [i for i in range(10)] + [i for i in range(10,58,4)] ]
    accur9 = accur9[:, [i for i in range(10)] + [i for i in range(10,58,4)] ]
    num_correct9 = num_correct9[:, [i for i in range(10)] + [i for i in range(10,58,4)] ]
    train_losses9 = train_losses9[:, [i for i in range(210)] + [i for i in range(210,1218,4)] ]
    # Test set 11 does four as many (but cheaper epochs)  So for plotting, we just take every other MGOpt epoch
    losses11, accur11, num_correct11, train_losses11 = \
            convert_data_for_plotting(losses11, accur11, num_correct11, train_losses11)
    losses11 = losses11[:, [i for i in range(10)] + [i for i in range(10,58,4)] ]
    accur11 = accur11[:, [i for i in range(10)] + [i for i in range(10,58,4)] ]
    num_correct11 = num_correct11[:, [i for i in range(10)] + [i for i in range(10,58,4)] ]
    train_losses11 = train_losses11[:, [i for i in range(210)] + [i for i in range(210,1218,4)] ]
    # Test set 12 does four as many (but cheaper epochs)  So for plotting, we just take every other MGOpt epoch
    losses12, accur12, num_correct12, train_losses12 = \
            convert_data_for_plotting(losses12, accur12, num_correct12, train_losses12)
    losses12 = losses12[:, [i for i in range(10)] + [i for i in range(10,58,4)] ]
    accur12 = accur12[:, [i for i in range(10)] + [i for i in range(10,58,4)] ]
    num_correct12 = num_correct12[:, [i for i in range(10)] + [i for i in range(10,58,4)] ]
    train_losses12 = train_losses12[:, [i for i in range(210)] + [i for i in range(210,1218,4)] ]


    # Total number of validation examples
    total_val_examples = 2000
    nrelax = 4.0    # number of mg/opt relaxations per iteration
    # Compute axes Scale axes to account for cheaper cost of the NI bootstrapping in the MGOpt solvers
    #                                  each NI step "costs" 1/nrelax              +       each MGOpt step costs "1"
    mgopt_val_xaxis = array( [ (1/nrelax)*k for k in range(MGOpt_start_val2)]     + [ (MGOpt_start_val2-1.0)/nrelax + k for k in range(1, losses2.shape[1] - MGOpt_start_val2 + 1) ] )
    mgopt_train_xaxis = array( [ (1/nrelax)*k for k in range(MGOpt_start_train2)] + [ (MGOpt_start_train2-1.0)/nrelax + k for k in range(1, train_losses2.shape[1] - MGOpt_start_train2 + 1) ] )
    # Now account for fact that the training results are printed out much more frequently 
    mgopt_train_xaxis = (max(mgopt_val_xaxis) / max(mgopt_train_xaxis)) * mgopt_train_xaxis
    
    # Do a simple linear scaling of the NI data (not quite accurate, but good enough) 
    NI_train_xaxis10 = arange(train_losses10.shape[1], dtype=float)
    NI_train_xaxis10 = (max(mgopt_val_xaxis) / max(NI_train_xaxis10)) * NI_train_xaxis10
    NI_val_xaxis10 = arange(losses10.shape[1], dtype=float)
    NI_val_xaxis10 = (max(mgopt_val_xaxis) / max(NI_val_xaxis10)) * NI_val_xaxis10
    #
    NI_train_xaxis1 = arange(train_losses1.shape[1], dtype=float)
    NI_train_xaxis1 = (max(mgopt_val_xaxis) / max(NI_train_xaxis1)) * NI_train_xaxis1
    NI_val_xaxis1 = arange(losses1.shape[1], dtype=float)
    NI_val_xaxis1 = (max(mgopt_val_xaxis) / max(NI_val_xaxis1)) * NI_val_xaxis1

    # Add one to all xaxis (work starts at 1)
    mgopt_val_xaxis += 1
    mgopt_train_xaxis += 1
    NI_train_xaxis10 += 1
    NI_val_xaxis10 += 1
    NI_train_xaxis1 += 1
    NI_val_xaxis1 += 1

    # Plot Losses
    fig1, ax1 = plt.subplots(1,1)
    # Plot NI losses, training and validation 
    #     Have to scale the x-axes because different numbers of epochs are done for Pure NI than MGOP
    plot_losses(ax1, losses1, train_losses1, NI_val_xaxis1, NI_train_xaxis1, colors[0], label='NI')
    plot_losses(ax1, losses10, train_losses10, NI_val_xaxis10, NI_train_xaxis10, colors[4], label='Adam')
    # Plot NI+MGOpt losses, training and validation
    #plot_losses(ax1, losses2, train_losses2, mgopt_val_xaxis, mgopt_train_xaxis, colors[1], label='NI+MGOpt, 2&5 Relax')
    #plot_losses(ax1, losses4, train_losses4, mgopt_val_xaxis, mgopt_train_xaxis, colors[2], label='NI+MGOpt, 0.1 Damp')
    #plot_losses(ax1, losses5, train_losses5, mgopt_val_xaxis, mgopt_train_xaxis, colors[3], label='NI+MGOpt, 0.01 Damp')
    #plot_losses(ax1, losses6, train_losses6, mgopt_val_xaxis, mgopt_train_xaxis, colors[5], label='NI+MGOpt, 0.000003 Damp')
    #plot_losses(ax1, losses7, train_losses7, mgopt_val_xaxis, mgopt_train_xaxis, colors[2], label='NI+MGOpt, WeakRelaxOnly')
    #plot_losses(ax1, losses8, train_losses8, mgopt_val_xaxis, mgopt_train_xaxis, colors[3], label='NI+MGOpt, Weak Relax&CGC, 0.01 Damp')
    plot_losses(ax1, losses9, train_losses9, mgopt_val_xaxis, mgopt_train_xaxis, colors[3], label='NI+MGOpt, 1 Relax')
    #plot_losses(ax1, losses11, train_losses11, mgopt_val_xaxis, mgopt_train_xaxis, colors[1], label='NI+MGOpt, 3-level, 1 Relax')
    plot_losses(ax1, losses12, train_losses12, mgopt_val_xaxis, mgopt_train_xaxis, colors[2], label='NI+MGOpt, 4-level, 1 Relax')
    #plot_losses(ax1, losses3, train_losses3, mgopt_val_xaxis, mgopt_train_xaxis, colors[0], label='NI+MGOpt+LocalRelax')
    # Labels and such
    ax1.set_xlabel('Work Units (4 Relaxations)', fontsize='large')
    ax1.set_ylabel('Loss', fontsize='large')
    ax1.legend(loc='lower left')
    plt.savefig('compare_losses.png', pad_inches=0.12, bbox_inches='tight', dpi=230)
    #

    # Plot number of missed validation test cases
    fig2, ax2 = plt.subplots(1,1)
    plot_validation_misclassified(ax2, total_val_examples-num_correct1, NI_val_xaxis1, colors[0], label='NI')
    plot_validation_misclassified(ax2, total_val_examples-num_correct10, NI_val_xaxis10, colors[4], label='Adam')
    #plot_validation_misclassified(ax2, total_val_examples-num_correct2, mgopt_val_xaxis, colors[1], label='NI+MGOpt, 2&5 Relax')
    #plot_validation_misclassified(ax2, total_val_examples-num_correct4, mgopt_val_xaxis, colors[2], label='NI+MGOpt, 0.1 Damp')
    #plot_validation_misclassified(ax2, total_val_examples-num_correct5, mgopt_val_xaxis, colors[3], label='NI+MGOpt, 0.01 Damp')
    #plot_validation_misclassified(ax2, total_val_examples-num_correct6, mgopt_val_xaxis, colors[5], label='NI+MGOpt, 0.000003 Damp')
    #plot_validation_misclassified(ax2, total_val_examples-num_correct7, mgopt_val_xaxis, colors[2], label='NI+MGOpt, WeakRelaxOnly')
    #plot_validation_misclassified(ax2, total_val_examples-num_correct8, mgopt_val_xaxis, colors[3], label='NI+MGOpt, Weak Relax&CGC, 0.01 Damp')
    plot_validation_misclassified(ax2, total_val_examples-num_correct9, mgopt_val_xaxis, colors[3], label='NI+MGOpt, 1 Relax')
    #plot_validation_misclassified(ax2, total_val_examples-num_correct11, mgopt_val_xaxis, colors[1], label='NI+MGOpt, 3-level, 1 Relax')
    plot_validation_misclassified(ax2, total_val_examples-num_correct12, mgopt_val_xaxis, colors[2], label='NI+MGOpt, 4-level, 1 Relax')
    #plot_validation_misclassified(ax2, total_val_examples-num_correct3, mgopt_val_xaxis, colors[0], label='NI+MGOpt+LocalRelax')
    # Labels and such
    ax2.set_xlabel('Work Units (4 Relaxations)', fontsize='large')
    ax2.set_ylabel('Accuracy\nNumber Incorrect Validation Examples (2000 total)', fontsize='large')
    ax2.legend(loc='upper right')
    plt.savefig('compare_accuracy.png', pad_inches=0.12, bbox_inches='tight', dpi=230)


# Turn on the 100% MNIST plots, 8 channels
if False:

    # test_results/simple_ls_67db6bbf/  test result for hash 67db6bbf using the defaults in that repo, nothing else, simple line-search
    losses1, train_losses1, accur1, num_correct1, MGOpt_start_val1, MGOpt_start_train1 = \
            grab_losses_acc_and_NI_MGOpt_transitions('test_results/simple_ls_be223a59d0/TB_NI.out')
    losses2, train_losses2, accur2, num_correct2, MGOpt_start_val2, MGOpt_start_train2 = \
            grab_losses_acc_and_NI_MGOpt_transitions('test_results/simple_ls_be223a59d0/TB_NI_onelevel_adam.out')
    losses3, train_losses3, accur3, num_correct3, MGOpt_start_val3, MGOpt_start_train3 = \
            grab_losses_acc_and_NI_MGOpt_transitions('test_results/simple_ls_be223a59d0/TB_NI_MGOpt_two_level.out')
    losses4, train_losses4, accur4, num_correct4, MGOpt_start_val4, MGOpt_start_train4 = \
            grab_losses_acc_and_NI_MGOpt_transitions('test_results/simple_ls_be223a59d0/TB_NI_MGOpt_three_levels.out')

    # Convert to nice for plotting arrays
    losses1, accur1, num_correct1, train_losses1 = \
            convert_data_for_plotting(losses1, accur1, num_correct1, train_losses1)
    losses2, accur2, num_correct2, train_losses2 = \
        convert_data_for_plotting(losses2, accur2, num_correct2, train_losses2)
    losses3, accur3, num_correct3, train_losses3 = \
        convert_data_for_plotting(losses3, accur3, num_correct3, train_losses3)
    losses4, accur4, num_correct4, train_losses4 = \
        convert_data_for_plotting(losses4, accur4, num_correct4, train_losses4)

    # Total number of validation examples
    total_val_examples = 10000
    nrelax = 3.0    # number of mg/opt relaxations per iteration
    # Compute axes Scale axes to account for cheaper cost of the NI bootstrapping in the MGOpt solvers
    #                                  each NI step "costs" 1/nrelax              +       each MGOpt step costs "1"
    mgopt_val_xaxis = array( [ (1/nrelax)*k for k in range(MGOpt_start_val3)]     + [ (MGOpt_start_val3-1.0)/nrelax + k for k in range(1, losses3.shape[1] - MGOpt_start_val3 + 1) ] )
    mgopt_train_xaxis = array( [ (1/nrelax)*k for k in range(MGOpt_start_train3)] + [ (MGOpt_start_train3-1.0)/nrelax + k for k in range(1, train_losses3.shape[1] - MGOpt_start_train3 + 1) ] )
    # Now account for fact that the training results are printed out much more frequently 
    mgopt_train_xaxis = (max(mgopt_val_xaxis) / max(mgopt_train_xaxis)) * mgopt_train_xaxis
    # Do a simple linear scaling of the NI data (not quite accurate, but good enough) 
    NI_train_xaxis1 = arange(train_losses1.shape[1], dtype=float)
    NI_train_xaxis1 = (max(mgopt_val_xaxis) / max(NI_train_xaxis1)) * NI_train_xaxis1
    NI_val_xaxis1 = arange(losses1.shape[1], dtype=float)
    NI_val_xaxis1 = (max(mgopt_val_xaxis) / max(NI_val_xaxis1)) * NI_val_xaxis1
    #
    NI_train_xaxis2 = arange(train_losses2.shape[1], dtype=float)
    NI_train_xaxis2 = (max(mgopt_val_xaxis) / max(NI_train_xaxis2)) * NI_train_xaxis2
    NI_val_xaxis2 = arange(losses2.shape[1], dtype=float)
    NI_val_xaxis2 = (max(mgopt_val_xaxis) / max(NI_val_xaxis2)) * NI_val_xaxis2

    # Plot Losses
    fig1, ax1 = plt.subplots(1,1)
    # Plot NI losses, training and validation 
    #     Have to scale the x-axes because different numbers of epochs are done for Pure NI than MGOP
    plot_losses(ax1, losses1, train_losses1, NI_val_xaxis1, NI_train_xaxis1, colors[0], label='NI')
    plot_losses(ax1, losses2, train_losses2, NI_val_xaxis2, NI_train_xaxis2, colors[1], label='Adam')
    # Plot NI+MGOpt losses, training and validation
    plot_losses(ax1, losses3, train_losses3, mgopt_val_xaxis, mgopt_train_xaxis, colors[2], label='NI+MGOpt, 2-level')
    plot_losses(ax1, losses4[:, 202:], train_losses4[:, 202:], mgopt_val_xaxis, mgopt_train_xaxis, colors[3], label='NI+MGOpt, 3-level') # skip the first 202 results (they are the 4-steps coarsest level)
    # Labels and such
    ax1.set_xlabel('Work Units (3 Relaxations)', fontsize='large')
    ax1.set_ylabel('Loss', fontsize='large')
    ax1.legend(loc='lower left')
    plt.savefig('compare_losses2.png', pad_inches=0.12, bbox_inches='tight', dpi=230)
    #

    # Plot number of missed validation test cases
    fig2, ax2 = plt.subplots(1,1)
    plot_validation_misclassified(ax2, total_val_examples-num_correct1, NI_val_xaxis1, colors[0], label='NI')
    plot_validation_misclassified(ax2, total_val_examples-num_correct2, NI_val_xaxis2, colors[1], label='Adam')
    #
    plot_validation_misclassified(ax2, total_val_examples-num_correct3, mgopt_val_xaxis, colors[2], label='NI+MGOpt, 2-level')
    plot_validation_misclassified(ax2, total_val_examples-num_correct4[:, 2:], mgopt_val_xaxis, colors[3], label='NI+MGOpt, 3-level')  # skip the first two results (they are the 4-steps coarsest level)
    # Labels and such
    ax2.set_xlabel('Work Units (3 Relaxations)', fontsize='large')
    ax2.set_ylabel('Accuracy\nNumber Incorrect Validation Examples (10000 total)', fontsize='large')
    ax2.legend(loc='upper right')
    plt.savefig('compare_accuracy2.png', pad_inches=0.12, bbox_inches='tight', dpi=230)


# Turn on the cheaper 100% MNIST plots, 4 channels
if False:

    # test_results/simple_ls_67db6bbf/  test result for hash 67db6bbf using the defaults in that repo, nothing else, simple line-search
    losses1, train_losses1, accur1, num_correct1, MGOpt_start_val1, MGOpt_start_train1 = \
            grab_losses_acc_and_NI_MGOpt_transitions('test_results/simple_ls_2ee9e50f31_4channel/TB_NI.out')
    losses2, train_losses2, accur2, num_correct2, MGOpt_start_val2, MGOpt_start_train2 = \
            grab_losses_acc_and_NI_MGOpt_transitions('test_results/simple_ls_2ee9e50f31_4channel/TB_NI_onelevel_adam.out')
    losses3, train_losses3, accur3, num_correct3, MGOpt_start_val3, MGOpt_start_train3 = \
            grab_losses_acc_and_NI_MGOpt_transitions('test_results/simple_ls_2ee9e50f31_4channel/TB_NI_MGOpt_twolevel_nomomentum.out')
    losses4, train_losses4, accur4, num_correct4, MGOpt_start_val4, MGOpt_start_train4 = \
            grab_losses_acc_and_NI_MGOpt_transitions('test_results/simple_ls_2ee9e50f31_4channel/TB_NI_MGOpt_twolevel_withmomentum_V10.out')
    #losses5, train_losses5, accur5, num_correct5, MGOpt_start_val5, MGOpt_start_train5 = \
    #        grab_losses_acc_and_NI_MGOpt_transitions('test_results/simple_ls_2ee9e50f31/..')
    #losses6, train_losses6, accur6, num_correct6, MGOpt_start_val6, MGOpt_start_train6 = \
    #        grab_losses_acc_and_NI_MGOpt_transitions('test_results/simple_ls_2ee9e50f31/..')
    #losses7, train_losses7, accur7, num_correct7, MGOpt_start_val7, MGOpt_start_train7 = \
    #        grab_losses_acc_and_NI_MGOpt_transitions('test_results/simple_ls_2ee9e50f31/..')

    # Convert to nice for plotting arrays
    losses1, accur1, num_correct1, train_losses1 = \
            convert_data_for_plotting(losses1, accur1, num_correct1, train_losses1)
    losses2, accur2, num_correct2, train_losses2 = \
        convert_data_for_plotting(losses2, accur2, num_correct2, train_losses2)
    losses3, accur3, num_correct3, train_losses3 = \
        convert_data_for_plotting(losses3, accur3, num_correct3, train_losses3)
    losses4, accur4, num_correct4, train_losses4 = \
        convert_data_for_plotting(losses4, accur4, num_correct4, train_losses4)
    #losses5, accur5, num_correct5, train_losses5 = \
    #    convert_data_for_plotting(losses5, accur5, num_correct5, train_losses5)
    #losses6, accur6, num_correct6, train_losses6 = \
    #    convert_data_for_plotting(losses6, accur6, num_correct6, train_losses6)
    #losses7, accur7, num_correct7, train_losses7 = \
    #    convert_data_for_plotting(losses7, accur7, num_correct7, train_losses7)

    # Total number of validation examples
    total_val_examples = 10000
    nrelax = 4.1    # number of mg/opt relaxations per iteration
    # Compute axes Scale axes to account for cheaper cost of the NI bootstrapping in the MGOpt solvers
    #                                  each NI step "costs" 1/nrelax              +       each MGOpt step costs "1"
    mgopt_val_xaxis3 = array( [ (1/nrelax)*k for k in range(MGOpt_start_val3)]     + [ (MGOpt_start_val3-1.0)/nrelax + k for k in range(1, losses3.shape[1] - MGOpt_start_val3 + 1) ] )
    mgopt_train_xaxis3 = array( [ (1/nrelax)*k for k in range(MGOpt_start_train3)] + [ (MGOpt_start_train3-1.0)/nrelax + k for k in range(1, train_losses3.shape[1] - MGOpt_start_train3 + 1) ] )
    # Now account for fact that the training results are printed out much more frequently 
    mgopt_train_xaxis3 = (max(mgopt_val_xaxis3) / max(mgopt_train_xaxis3)) * mgopt_train_xaxis3
    ## 
    mgopt_val_xaxis4 = array( [ (1/nrelax)*k for k in range(MGOpt_start_val4)]     + [ (MGOpt_start_val4-1.0)/nrelax + k for k in range(1, losses4.shape[1] - MGOpt_start_val4 + 1) ] )
    mgopt_train_xaxis4 = array( [ (1/nrelax)*k for k in range(MGOpt_start_train4)] + [ (MGOpt_start_train4-1.0)/nrelax + k for k in range(1, train_losses4.shape[1] - MGOpt_start_train4 + 1) ] )
    # Now account for fact that the training results are printed out much more frequently 
    mgopt_train_xaxis4 = (max(mgopt_val_xaxis4) / max(mgopt_train_xaxis4)) * mgopt_train_xaxis4
    # Now, scale both relative to dataset 3
    mgopt_train_xaxis4 = (max(mgopt_train_xaxis3) / max(mgopt_train_xaxis4)) * mgopt_train_xaxis4
    mgopt_val_xaxis4 = (max(mgopt_val_xaxis3) / max(mgopt_val_xaxis4)) * mgopt_val_xaxis4
    ##
    # Do a simple linear scaling of the NI data (not quite accurate, but good enough) 
    NI_train_xaxis1 = arange(train_losses1.shape[1], dtype=float)
    NI_train_xaxis1 = (max(mgopt_val_xaxis3) / max(NI_train_xaxis1)) * NI_train_xaxis1
    NI_val_xaxis1 = arange(losses1.shape[1], dtype=float)
    NI_val_xaxis1 = (max(mgopt_val_xaxis3) / max(NI_val_xaxis1)) * NI_val_xaxis1
    #
    NI_train_xaxis2 = arange(train_losses2.shape[1], dtype=float)
    NI_train_xaxis2 = (max(mgopt_val_xaxis3) / max(NI_train_xaxis2)) * NI_train_xaxis2
    NI_val_xaxis2 = arange(losses2.shape[1], dtype=float)
    NI_val_xaxis2 = (max(mgopt_val_xaxis3) / max(NI_val_xaxis2)) * NI_val_xaxis2

    # Plot Losses
    fig1, ax1 = plt.subplots(1,1)
    # Plot NI losses, training and validation 
    #     Have to scale the x-axes because different numbers of epochs are done for Pure NI than MGOP
    plot_losses(ax1, losses1, train_losses1, NI_val_xaxis1, NI_train_xaxis1, colors[0], label='NI')
    plot_losses(ax1, losses2, train_losses2, NI_val_xaxis2, NI_train_xaxis2, colors[1], label='Adam')
    # Plot NI+MGOpt losses, training and validation
    plot_losses(ax1, losses3, train_losses3, mgopt_val_xaxis3, mgopt_train_xaxis3, colors[2], label='NI+MGOpt, 2-level, No Momentum')
    #import pdb; pdb.set_trace()
    plot_losses(ax1, losses4, train_losses4, mgopt_val_xaxis4, mgopt_train_xaxis4, colors[3], label='NI+MGOpt, 2-level, W/ Momentum') 
    # Labels and such
    ax1.set_xlabel('Work Units (~4.1 Relaxations)', fontsize='large')
    ax1.set_ylabel('Loss', fontsize='large')
    ax1.legend(loc='lower left')
    plt.savefig('compare_losses2.png', pad_inches=0.12, bbox_inches='tight', dpi=230)
    #

    # Plot number of missed validation test cases
    fig2, ax2 = plt.subplots(1,1)
    plot_validation_misclassified(ax2, total_val_examples-num_correct1, NI_val_xaxis1, colors[0], label='NI')
    plot_validation_misclassified(ax2, total_val_examples-num_correct2, NI_val_xaxis2, colors[1], label='Adam')
    #
    plot_validation_misclassified(ax2, total_val_examples-num_correct3, mgopt_val_xaxis3, colors[2], label='NI+MGOpt, 2-level, No Momentum')
    plot_validation_misclassified(ax2, total_val_examples-num_correct4, mgopt_val_xaxis4, colors[3], label='NI+MGOpt, 2-level, W/ Momentum')  
    # Labels and such
    ax2.set_xlabel('Work Units (~4.1 Relaxations)', fontsize='large')
    ax2.set_ylabel('Accuracy\nNumber Incorrect Validation Examples (10000 total)', fontsize='large')
    ax2.legend(loc='upper right')
    plt.savefig('compare_accuracy2.png', pad_inches=0.12, bbox_inches='tight', dpi=230)



# Turn on the 100% MNIST plots, 8 channels, slightly newer repository version (slightly different results
if True:

    # test_results/simple_ls_67db6bbf/  test result for hash 67db6bbf using the defaults in that repo, nothing else, simple line-search
    losses1, train_losses1, accur1, num_correct1, MGOpt_start_val1, MGOpt_start_train1 = \
            grab_losses_acc_and_NI_MGOpt_transitions('test_results/simple_ls_2ee9e50f31_8channel/TB_NI_wmomentum.out')
    losses2, train_losses2, accur2, num_correct2, MGOpt_start_val2, MGOpt_start_train2 = \
            grab_losses_acc_and_NI_MGOpt_transitions('test_results/simple_ls_2ee9e50f31_8channel/TB_NI_singlelevel_adam_wmomentum.out')
    losses3, train_losses3, accur3, num_correct3, MGOpt_start_val3, MGOpt_start_train3 = \
            grab_losses_acc_and_NI_MGOpt_transitions('test_results/simple_ls_2ee9e50f31_8channel/TB_NI_MGOPT_two_level_V11_woutmomentum_QUARTZ.out')
    losses4, train_losses4, accur4, num_correct4, MGOpt_start_val4, MGOpt_start_train4 = \
            grab_losses_acc_and_NI_MGOpt_transitions('test_results/simple_ls_2ee9e50f31_8channel/TB_NI_MGOPT_two_level_V11_wmomentum.out')
    losses5, train_losses5, accur5, num_correct5, MGOpt_start_val5, MGOpt_start_train5 = \
            grab_losses_acc_and_NI_MGOpt_transitions('test_results/simple_ls_2ee9e50f31_8channel/TB_NI_MGOPT_two_level_V01_woutmomentum.out')
    losses6, train_losses6, accur6, num_correct6, MGOpt_start_val6, MGOpt_start_train6 = \
            grab_losses_acc_and_NI_MGOpt_transitions('test_results/simple_ls_2ee9e50f31_8channel/TB_NI_MGOPT_two_level_V01_wmomentum.out')
    losses7, train_losses7, accur7, num_correct7, MGOpt_start_val7, MGOpt_start_train7 = \
            grab_losses_acc_and_NI_MGOpt_transitions('test_results/simple_ls_2ee9e50f31_8channel/TB_NI_MGOPT_two_level_V01_wmomentum_001_fixed_damping.out')
    losses8, train_losses8, accur8, num_correct8, MGOpt_start_val8, MGOpt_start_train8 = \
            grab_losses_acc_and_NI_MGOpt_transitions('test_results/simple_ls_2ee9e50f31_8channel/TB_NI_MGOPT_two_level_V01_wmomentum_armijo0001.out')
  ##losses9, train_losses9, accur9, num_correct9, MGOpt_start_val9, MGOpt_start_train9 = \
  ##        grab_losses_acc_and_NI_MGOpt_transitions('test_results/simple_ls_2ee9e50f31_8channel/TB_NI_MGOPT_two_level_V11_woutmomentum_armijo0001.out')
  ##losses10, train_losses10, accur10, num_correct10, MGOpt_start_val10, MGOpt_start_train10 = \
  ##        grab_losses_acc_and_NI_MGOpt_transitions('test_results/simple_ls_2ee9e50f31_8channel/TB_MGOPT_two_level_V01_wmomentum_armijo0001.out')
  ##losses11, train_losses11, accur11, num_correct11, MGOpt_start_val11, MGOpt_start_train11 = \
  ##        grab_losses_acc_and_NI_MGOpt_transitions('test_results/simple_ls_2ee9e50f31_8channel/TB_MGOPT_two_level_V11_woutmomentum_armijo0001.out')

    # Convert to nice for plotting arrays
    losses1, accur1, num_correct1, train_losses1 = \
            convert_data_for_plotting(losses1, accur1, num_correct1, train_losses1)
    losses2, accur2, num_correct2, train_losses2 = \
        convert_data_for_plotting(losses2, accur2, num_correct2, train_losses2)
    losses3, accur3, num_correct3, train_losses3 = \
        convert_data_for_plotting(losses3, accur3, num_correct3, train_losses3)
    losses4, accur4, num_correct4, train_losses4 = \
        convert_data_for_plotting(losses4, accur4, num_correct4, train_losses4)
    losses5, accur5, num_correct5, train_losses5 = \
        convert_data_for_plotting(losses5, accur5, num_correct5, train_losses5)
    losses6, accur6, num_correct6, train_losses6 = \
        convert_data_for_plotting(losses6, accur6, num_correct6, train_losses6)
    losses7, accur7, num_correct7, train_losses7 = \
        convert_data_for_plotting(losses7, accur7, num_correct7, train_losses7)
    losses8, accur8, num_correct8, train_losses8 = \
        convert_data_for_plotting(losses8, accur8, num_correct8, train_losses8)
  ##losses9, accur9, num_correct9, train_losses9 = \
  ##    convert_data_for_plotting(losses9, accur9, num_correct9, train_losses9)
  ##losses10, accur10, num_correct10, train_losses10 = \
  ##    convert_data_for_plotting(losses10, accur10, num_correct10, train_losses10)
  ##losses11, accur11, num_correct11, train_losses11 = \
  ##    convert_data_for_plotting(losses11, accur11, num_correct11, train_losses11)

    # Total number of validation examples
    total_val_examples = 10000
    nrelax = 4.1    # number of mg/opt relaxations per iteration
    # Compute axes Scale axes to account for cheaper cost of the NI bootstrapping in the MGOpt solvers
    #                                  each NI step "costs" 1/nrelax              +       each MGOpt step costs "1"
    mgopt_val_xaxis3 = array( [ (1/nrelax)*k for k in range(MGOpt_start_val3)]     + [ (MGOpt_start_val3-1.0)/nrelax + k for k in range(1, losses3.shape[1] - MGOpt_start_val3 + 1) ] )
    mgopt_train_xaxis3 = array( [ (1/nrelax)*k for k in range(MGOpt_start_train3)] + [ (MGOpt_start_train3-1.0)/nrelax + k for k in range(1, train_losses3.shape[1] - MGOpt_start_train3 + 1) ] )
    # Now account for fact that the training results are printed out much more frequently 
    mgopt_train_xaxis3 = (max(mgopt_val_xaxis3) / max(mgopt_train_xaxis3)) * mgopt_train_xaxis3
    ###
    mgopt_val_xaxis4 = array( [ (1/nrelax)*k for k in range(MGOpt_start_val4)]     + [ (MGOpt_start_val4-1.0)/nrelax + k for k in range(1, losses4.shape[1] - MGOpt_start_val4 + 1) ] )
    mgopt_train_xaxis4 = array( [ (1/nrelax)*k for k in range(MGOpt_start_train4)] + [ (MGOpt_start_train4-1.0)/nrelax + k for k in range(1, train_losses4.shape[1] - MGOpt_start_train4 + 1) ] )
    # Now account for fact that the training results are printed out much more frequently 
    mgopt_train_xaxis4 = (max(mgopt_val_xaxis4) / max(mgopt_train_xaxis4)) * mgopt_train_xaxis4
    # Now, scale both relative to dataset 3
    mgopt_train_xaxis4 = (max(mgopt_train_xaxis3) / max(mgopt_train_xaxis4)) * mgopt_train_xaxis4
    mgopt_val_xaxis4 = (max(mgopt_val_xaxis3) / max(mgopt_val_xaxis4)) * mgopt_val_xaxis4
    ###
    mgopt_val_xaxis5 = array( [ (1/nrelax)*k for k in range(MGOpt_start_val5)]     + [ (MGOpt_start_val5-1.0)/nrelax + k for k in range(1, losses5.shape[1] - MGOpt_start_val5 + 1) ] )
    mgopt_train_xaxis5 = array( [ (1/nrelax)*k for k in range(MGOpt_start_train5)] + [ (MGOpt_start_train5-1.0)/nrelax + k for k in range(1, train_losses5.shape[1] - MGOpt_start_train5 + 1) ] )
    # Now account for fact that the training results are printed out much more frequently 
    mgopt_train_xaxis5 = (max(mgopt_val_xaxis5) / max(mgopt_train_xaxis5)) * mgopt_train_xaxis5
    # Now, scale both relative to dataset 3
    mgopt_train_xaxis5 = (max(mgopt_train_xaxis3) / max(mgopt_train_xaxis5)) * mgopt_train_xaxis5
    mgopt_val_xaxis5 = (max(mgopt_val_xaxis3) / max(mgopt_val_xaxis5)) * mgopt_val_xaxis5
    ###
    # Do a simple linear scaling of the NI data (not quite accurate, but good enough) 
    NI_train_xaxis1 = arange(train_losses1.shape[1], dtype=float)
    NI_train_xaxis1 = (max(mgopt_val_xaxis3) / max(NI_train_xaxis1)) * NI_train_xaxis1
    NI_val_xaxis1 = arange(losses1.shape[1], dtype=float)
    NI_val_xaxis1 = (max(mgopt_val_xaxis3) / max(NI_val_xaxis1)) * NI_val_xaxis1
    #
    NI_train_xaxis2 = arange(train_losses2.shape[1], dtype=float)
    NI_train_xaxis2 = (max(mgopt_val_xaxis3) / max(NI_train_xaxis2)) * NI_train_xaxis2
    NI_val_xaxis2 = arange(losses2.shape[1], dtype=float)
    NI_val_xaxis2 = (max(mgopt_val_xaxis3) / max(NI_val_xaxis2)) * NI_val_xaxis2

    # Plot Losses
    fig1, ax1 = plt.subplots(1,1)
    # Plot NI losses, training and validation 
    #     Have to scale the x-axes because different numbers of epochs are done for Pure NI than MGOP
    plot_losses(ax1, losses1, train_losses1, NI_val_xaxis1, NI_train_xaxis1, colors[0], label='NI')
    plot_losses(ax1, losses2, train_losses2, NI_val_xaxis2, NI_train_xaxis2, colors[1], label='Adam')
    # Plot NI+MGOpt losses, training and validation
    plot_losses(ax1, losses3, train_losses3, mgopt_val_xaxis3, mgopt_train_xaxis3, colors[2], label='NI+MGOpt V(1,1), 2-level, No Momentum')
    plot_losses(ax1, losses4, train_losses4, mgopt_val_xaxis4, mgopt_train_xaxis4, colors[3], label='NI+MGOpt V(1,1), 2-level, W/ Momentum') 
    plot_losses(ax1, losses5, train_losses5, mgopt_val_xaxis5, mgopt_train_xaxis5, colors[4], label='NI+MGOpt V(0,1), 2-level, No Momentum')
    plot_losses(ax1, losses6, train_losses6, mgopt_val_xaxis5, mgopt_train_xaxis5, colors[5], label='NI+MGOpt V(0,1), 2-level, W/ Momentum') 
    #plot_losses(ax1, losses7, train_losses7, mgopt_val_xaxis5, mgopt_train_xaxis5, colors[6], label='NI+MGOpt V(0,1), 2-level, W/ Momentum, 001 Damping') 
    plot_losses(ax1, losses8, train_losses8, mgopt_val_xaxis5, mgopt_train_xaxis5, colors[6], label='NI+MGOpt V(0,1), 2-level, W/ Momentum, Armijo') 
    # Labels and such
    ax1.set_xlabel('Work Units (~4.1 Relaxations)', fontsize='large')
    ax1.set_ylabel('Loss', fontsize='large')
    ax1.legend(loc='lower left')
    plt.savefig('compare_losses3.png', pad_inches=0.12, bbox_inches='tight', dpi=230)
    #

    # Plot number of missed validation test cases
    fig2, ax2 = plt.subplots(1,1)
    plot_validation_misclassified(ax2, total_val_examples-num_correct1, NI_val_xaxis1, colors[0], label='NI')
    plot_validation_misclassified(ax2, total_val_examples-num_correct2, NI_val_xaxis2, colors[1], label='Adam')
    #
    plot_validation_misclassified(ax2, total_val_examples-num_correct3, mgopt_val_xaxis3, colors[2], label='NI+MGOpt V(1,1), 2-level, No Momentum')
    plot_validation_misclassified(ax2, total_val_examples-num_correct4, mgopt_val_xaxis4, colors[3], label='NI+MGOpt V(1,1), 2-level, W/ Momentum') 
    plot_validation_misclassified(ax2, total_val_examples-num_correct5, mgopt_val_xaxis5, colors[4], label='NI+MGOpt V(0,1), 2-level, No Momentum')
    plot_validation_misclassified(ax2, total_val_examples-num_correct6, mgopt_val_xaxis5, colors[5], label='NI+MGOpt V(0,1), 2-level, W/ Momentum') 
    #plot_validation_misclassified(ax2, total_val_examples-num_correct7, mgopt_val_xaxis5, colors[6], label='NI+MGOpt V(0,1), 2-level, W/ Momentum, 001 Damping') 
    plot_validation_misclassified(ax2, total_val_examples-num_correct8, mgopt_val_xaxis5, colors[6], label='NI+MGOpt V(0,1), 2-level, W/ Momentum, Armijo') 
    # Labels and such
    ax2.set_xlabel('Work Units (~4.1 Relaxations)', fontsize='large')
    ax2.set_ylabel('Accuracy\nNumber Incorrect Validation Examples (10000 total)', fontsize='large')
    ax2.legend(loc='upper right')
    plt.savefig('compare_number_correct3.png', pad_inches=0.12, bbox_inches='tight', dpi=230)

    # Plot accuracy percentages 
    fig2, ax2 = plt.subplots(1,1)
    plot_accur(ax2, accur1, NI_val_xaxis1, colors[0], label='NI')
    plot_accur(ax2, accur2, NI_val_xaxis2, colors[1], label='Adam')
    #       
    plot_accur(ax2, accur3, mgopt_val_xaxis3, colors[2], label='NI+MGOpt V(1,1), 2-level, No Momentum')
    plot_accur(ax2, accur4, mgopt_val_xaxis4, colors[3], label='NI+MGOpt V(1,1), 2-level, W/ Momentum') 
    plot_accur(ax2, accur5, mgopt_val_xaxis5, colors[4], label='NI+MGOpt V(0,1), 2-level, No Momentum')
    plot_accur(ax2, accur6, mgopt_val_xaxis5, colors[5], label='NI+MGOpt V(0,1), 2-level, W/ Momentum') 
    #plot_accur(ax2, accur7, mgopt_val_xaxis5, colors[6], label='NI+MGOpt V(0,1), 2-level, W/ Momentum, 001 Damping') 
    plot_accur(ax2, accur8, mgopt_val_xaxis5, colors[6], label='NI+MGOpt V(0,1), 2-level, W/ Momentum, Armijo') 
    # Labels and such
    ax2.set_xlabel('Work Units (~4.1 Relaxations)', fontsize='large')
    ax2.set_ylabel('Test Set Accuracy', fontsize='large')
    ax2.legend(loc='lower right')
    plt.savefig('compare_accuracy3.png', pad_inches=0.12, bbox_inches='tight', dpi=230)








