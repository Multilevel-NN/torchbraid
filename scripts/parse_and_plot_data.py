import matplotlib.pyplot as plt
import numpy as np


class run:
  def __init__(self, procs, path_torchbraid, paths_xbraid, label):
    self.procs = procs
    self.path_torchbraid = path_torchbraid
    self.paths_xbraid = paths_xbraid
    self.res_torchbraid = self.parse_torchbraid_timings(file=self.path_torchbraid)
    self.res_xbraid = self.parse_xbraid_file(braid_timings=self.paths_xbraid)
    self.label = label
    self.fwd_time_braid = {}
    self.bwd_time_braid = {}
    self.fwd_time_torchbraid = {}
    self.bwd_time_torchbraid = {}
    self.max_fwd_tb = 0
    self.max_bwd_tb = 0
    self.max_fwd_rb = 0
    self.max_bwd_rb = 0
    self.max_fwd_xb = 0
    self.max_bwd_xb = 0
    self.runtimes()

  def runtimes(self):
    for i in range(self.procs):
      counter_fwd_torchbraid = 0
      counter_bwd_torchbraid = 0
      counter_fwd_runBraid = 0
      counter_bwd_runBraid = 0
      counter_fwd_xbraid = 0
      counter_bwd_xbraid = 0
      for key, value in self.res_torchbraid[i]['fwd'].items():
        if key != 'runBraid':
          counter_fwd_torchbraid += value['total']
          if key in self.fwd_time_torchbraid:
            self.fwd_time_torchbraid[key].append(value['total'])
          else:
            self.fwd_time_torchbraid[key] = [value['total']]
        else:
          counter_fwd_runBraid = value['total']
      for key, value in self.res_torchbraid[i]['bwd'].items():
        if key != 'runBraid':
          counter_bwd_torchbraid += value['total']
          if key in self.bwd_time_torchbraid:
            self.bwd_time_torchbraid[key].append(value['total'])
          else:
            self.bwd_time_torchbraid[key] = [value['total']]
        else:
          counter_bwd_runBraid = value['total']

      for key, value in self.res_xbraid[i]['fwd'].items():
        counter_fwd_xbraid += value
        if key in self.fwd_time_braid:
          self.fwd_time_braid[key].append(value)
        else:
          self.fwd_time_braid[key] = [value]

      for key, value in self.res_xbraid[i]['bwd'].items():
        counter_bwd_xbraid += value
        if key in self.bwd_time_braid:
          self.bwd_time_braid[key].append(value)
        else:
          self.bwd_time_braid[key] = [value]

      if counter_fwd_runBraid > self.max_fwd_rb:
        self.max_fwd_rb = counter_fwd_runBraid
      if counter_bwd_runBraid > self.max_bwd_rb:
        self.max_bwd_rb = counter_bwd_runBraid
      if counter_fwd_torchbraid > self.max_fwd_tb:
        self.max_fwd_tb = counter_fwd_torchbraid
      if counter_bwd_torchbraid > self.max_bwd_tb:
        self.max_bwd_tb = counter_bwd_torchbraid
      if counter_fwd_xbraid > self.max_fwd_xb:
        self.max_fwd_xb = counter_fwd_xbraid
      if counter_bwd_xbraid > self.max_bwd_xb:
        self.max_bwd_xb = counter_bwd_xbraid

    self.fwd_time_braid['coarse_solve'] = np.array(self.fwd_time_braid['coarse_solve']) - np.array(
      self.fwd_time_braid['MPI_wait_coarse'])
    self.bwd_time_braid['coarse_solve'] = np.array(self.bwd_time_braid['coarse_solve']) - np.array(
      self.bwd_time_braid['MPI_wait_coarse'])

  def plot_figure(self):
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    self.plot(ax1=ax1, ax2=ax2)
    fig.suptitle(self.label, fontsize=14, y=-0.01)
    plt.show()

  def plot_times(self, ind, width, colors, dict, ax):
    first = True
    bars = [0 for item in range(len(ind))]
    keys = ['drive_init', 'coarse_solve', 'step', 'clone', 'free', 'sum', 'norm', 'bufsize', 'bufpack', 'bufunpack',
            'access', 'MPI_recv', 'MPI_send', 'MPI_wait', 'MPI_wait_coarse', 'getUVector']
    for key in keys:
      if key in dict:
        ax.bar(ind, dict[key], bottom=bars, width=width, label=key, color=colors[key])
        bars = np.add(bars, dict[key]).tolist()
    return max(bars)

  def plot(self, ax1, ax2, legend=True):
    run_braid_fwd = []
    run_braid_bwd = []
    width = 0.8

    colors = {'drive_init': '#3182bd',
              'coarse_solve': 'yellow',
              'step': '#9ecae1',
              'init': '#c6dbef',
              'clone': '#e6550d',
              'free': '#fd8d3c',
              'sum': '#fdae6b',
              'norm': '#fdd0a2',
              'bufsize': '#31a354',
              'bufpack': '#74c476',
              'bufunpack': '#a1d99b',
              'access': '#c7e9c0',
              'sync': '#756bb1',
              'residual': '#9e9ac8',
              'getUVector': '#bcbddc',
              'srefine': '#dadaeb',
              'MPI_recv': '#636363',
              'MPI_send': '#969696',
              'MPI_wait': 'purple',
              'MPI_wait_coarse': '#d9d9d9'}

    procs = len(self.res_torchbraid)
    for i in range(procs):
      run_braid_fwd.append(self.res_torchbraid[i]['fwd']['runBraid']['total'])
      run_braid_bwd.append(self.res_torchbraid[i]['bwd']['runBraid']['total'])

    ind = np.arange(len(run_braid_fwd))
    ax1.bar(ind, run_braid_fwd, width, label='runBraid (torchbraid)', color="None", edgecolor='black')
    ax2.bar(ind, run_braid_bwd, width, label='runBraid (torchbraid)', color="None", edgecolor='black')

    # Plot timings
    fwd_max = self.plot_times(ind=ind, width=width, colors=colors, dict=self.fwd_time_braid, ax=ax1)
    bwd_max = self.plot_times(ind=ind, width=width, colors=colors, dict=self.bwd_time_braid, ax=ax2)

    # Plot settings
    if legend:
      ax1.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3)
    ax1.set_xticks(ind + .5 * width)
    ax1.set_xticks(ind)
    ax1.set_xticklabels(ind)
    fwd_max = max(max(run_braid_fwd), fwd_max)
    bwd_max = max(max(run_braid_bwd), bwd_max)
    ax1.set_title('Forward')
    ax2.set_title('Backward')
    ax1.set_ylabel('runtime (s)')
    ax2.set_ylabel('runtime (s)')

  def parse_torchbraid_timings(self, file: str) -> dict:
    """
    Parses the Torchbraid timing information from the Torchbraid output file
    """
    results = {}
    with open(file) as f:
      lines = f.readlines()
      splits = [line.split() for line in lines]
      for i in range(len(splits)):
        tmp = splits[i]
        if len(tmp) > 0 and tmp[0] == 'Using':
          if tmp[1].startswith('SerialNet'):
            net = 'SerialNet'
          elif tmp[1].startswith('ParallelNet'):
            net = 'ParallelNet'
        if len(tmp) > 0 and tmp[0] == '***' and tmp[1] == 'Proc':
          proc = int(tmp[3])
          results[proc] = {}
          results[proc] = {'bwd': self.create_empty_dict('torchbraid'),
                           'fwd': self.create_empty_dict('torchbraid')}
          j = i + 3
          while len(splits[j]) != 0:
            results_split = splits[j]
            op = results_split[0].split('::')[1]
            count = int(results_split[2])
            total = float(results_split[4])
            mean = float(results_split[6])
            stdev = float(results_split[8])
            sub_op = None
            if results_split[0].startswith('BckWD'):
              type = 'bwd'
            elif results_split[0].startswith('ForWD'):
              type = 'fwd'
            else:
              raise Exception('unknown WD')
            op_split = op.split('-')
            if len(op_split) > 1:
              op = op_split[0]
              sub_op = op_split[1]
            if sub_op is None:
              results[proc][type][op]['count'] = count
              results[proc][type][op]['total'] = total
              results[proc][type][op]['mean'] = mean
              results[proc][type][op]['stdev'] = stdev
            else:
              results[proc][type][op]['subops'][sub_op] = {'count': count,
                                                           'total': total,
                                                           'mean': mean,
                                                           'stdev': stdev}
            j = j + 1
      return results

  def parse_xbraid_file(self, braid_timings) -> dict:
    """
    Parses all braid_timing files
    """
    results = {}
    for path in braid_timings:
      file = path.split('/')[-1]
      proc = int(file[-8:-4])
      type = file.split('_')[1]
      if proc not in results:
        results[proc] = {'bwd': self.create_empty_dict('xbraid'), 'fwd': self.create_empty_dict('xbraid')}
      with open(path) as f:
        lines = f.readlines()
        for line in lines:
          tmp = line.split()
          if len(tmp) == 0 or tmp[0] == 'Timings':
            continue
          op = tmp[0]
          if op == 'spatialnorm':
            op = 'norm'
          if op == 'coarse':
            results[proc][type]['coarse_solve'] = float(tmp[2])
          else:
            results[proc][type][op] = float(tmp[1])
    return results

  def create_empty_dict(self, type: str) -> dict:
    """
    Create dictionary to record information
    """
    if type == 'torchbraid':
      tmp = {'access': {'count': 0, 'total': 0, 'mean': 0, 'stdev': 0, 'subops': {}},
             'bufsize': {'count': 0, 'total': 0, 'mean': 0, 'stdev': 0, 'subops': {}},
             'bufpack': {'count': 0, 'total': 0, 'mean': 0, 'stdev': 0, 'subops': {}},
             'clone': {'count': 0, 'total': 0, 'mean': 0, 'stdev': 0, 'subops': {}},
             'free': {'count': 0, 'total': 0, 'mean': 0, 'stdev': 0, 'subops': {}},
             'getUVector': {'count': 0, 'total': 0, 'mean': 0, 'stdev': 0, 'subops': {}},
             'norm': {'count': 0, 'total': 0, 'mean': 0, 'stdev': 0, 'subops': {}},
             'runBraid': {'count': 0, 'total': 0, 'mean': 0, 'stdev': 0, 'subops': {}},
             'step': {'count': 0, 'total': 0, 'mean': 0, 'stdev': 0, 'subops': {}},
             'sum': {'count': 0, 'total': 0, 'mean': 0, 'stdev': 0, 'subops': {}},
             'bufunpack': {'count': 0, 'total': 0, 'mean': 0, 'stdev': 0, 'subops': {}}}
    elif type == 'xbraid':
      tmp = {'drive_init': 0,
             'coarse_solve': 0,
             'step': 0,
             'init': 0,
             'clone': 0,
             'free': 0,
             'sum': 0,
             'norm': 0,
             'bufsize': 0,
             'bufpack': 0,
             'bufunpack': 0,
             'access': 0,
             'sync': 0,
             'residual': 0,
             'scoarsen': 0,
             'srefine': 0,
             'MPI_recv': 0,
             'MPI_send': 0,
             'MPI_wait': 0,
             'MPI_wait_coarse': 0
             }
    else:
      raise Exception('unknown type')
    return tmp


def plot_overview(runs, save=None):
  fig, axs = plt.subplots(nrows=2, ncols=len(runs), sharex=True, figsize=(15, 8))
  max_fwd = max([item.max_fwd_xb for item in runs])
  max_bwd = max([item.max_bwd_xb for item in runs])
  procs = max([item.procs for item in runs])
  for i in range(len(runs)):
    runs[i].plot(ax1=axs[0][i], ax2=axs[1][i], legend=False)
    axs[0][i].set_ylim(top=max_fwd*1.3)
    axs[1][i].set_ylim(top=max_bwd*1.3)
  axs[0][0].set_xlim(right=procs)
  handles, labels = axs[0][0].get_legend_handles_labels()
  fig.legend(handles, labels, loc='upper center', ncol=len(runs)*2)
  if save is not None:
    plt.savefig(save, format='pdf', bbox_inches='tight')
  else:
    plt.show()


if __name__ == '__main__':
  path = 'path/to/results/'
  p1 = run(procs=1,
          path_torchbraid=path + 'script_1.out',
          paths_xbraid=[path + 'b_fwd_s_16_c_64_bs_200_p_1_0000.txt',
                        path + 'b_bwd_s_16_c_64_bs_200_p_1_0000.txt'],
          label='P=1')
  p2 = run(procs=2,
          path_torchbraid=path + 'script_2.out',
          paths_xbraid=[path + 'b_fwd_s_16_c_64_bs_200_p_2_0000.txt',
                        path + 'b_fwd_s_16_c_64_bs_200_p_2_0001.txt',
                        path + 'b_bwd_s_16_c_64_bs_200_p_2_0000.txt',
                        path + 'b_bwd_s_16_c_64_bs_200_p_2_0001.txt'],
          label='P=2')
  p4 = run(procs=4,
          path_torchbraid=path + 'script_4.out',
          paths_xbraid=[path + 'b_fwd_s_16_c_64_bs_200_p_4_0000.txt',
                        path + 'b_fwd_s_16_c_64_bs_200_p_4_0001.txt',
                        path + 'b_fwd_s_16_c_64_bs_200_p_4_0002.txt',
                        path + 'b_fwd_s_16_c_64_bs_200_p_4_0003.txt',
                        path + 'b_bwd_s_16_c_64_bs_200_p_4_0000.txt',
                        path + 'b_bwd_s_16_c_64_bs_200_p_4_0001.txt',
                        path + 'b_bwd_s_16_c_64_bs_200_p_4_0002.txt',
                        path + 'b_bwd_s_16_c_64_bs_200_p_4_0003.txt'],
          label='P=4')
  #p1.plot_figure()
  #p2.plot_figure()
  #p4.plot_figure()
  plot_overview(runs=[p1, p2, p4])
