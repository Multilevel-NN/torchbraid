import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
from matplotlib.lines import Line2D

COLOR_LIST = ['#4c72b0', '#dd8452', '#55a868', '#c44e52', '#8172b3', '#937860', '#da8bc3', '#8c8c8c', '#ccb974',
              '#64b5cd', '#818d6d', '#7f0c17', '#c4ddb2', '#2ab414', '#f98131', '#08786d', '#142840',
              '#d065b5', '#a73307']

colors = {
  'my_step_0': COLOR_LIST[0],
  'my_step_1': COLOR_LIST[1],
  'my_step_2': COLOR_LIST[2],
  'my_step_3': COLOR_LIST[3],
  'my_step_4': COLOR_LIST[4],
  'my_bufunpack_0': COLOR_LIST[5],
  'my_bufunpack_1': COLOR_LIST[6],
  'my_bufunpack_2': COLOR_LIST[7],
  'my_bufunpack_3': COLOR_LIST[8],
  'my_bufunpack_4': COLOR_LIST[9],
  'my_bufalloc': COLOR_LIST[10],
  'initializeStates': COLOR_LIST[11],
  'myClone': COLOR_LIST[12],
  'my_sum': COLOR_LIST[13],
  'my_free': COLOR_LIST[14],
  'my_buffree': COLOR_LIST[15],
}

used_ops = {}


def parseOutputFile(path, procs, run):
  rectangles = []
  rectangles_r = []
  run_counter = [1 for _ in range(procs)]
  min_time = [0 for _ in range(procs)]
  max_time = [0 for _ in range(procs)]
  type = 'None'
  with open(path) as file:
    for line in file:
      if line.strip().startswith("Model"):
        content = line.rstrip().replace(':', '|').split('|')
        op = content[8].strip()
        rank = int(content[2])
        t_start = float(content[4])
        t_stop = float(content[6])
        type = content[10]
        if op == 'runBraid':
          if run == run_counter[rank]:
            rectangles_r.append(["", mpatch.Rectangle((t_start, rank - .25), t_stop - t_start,
                                                      .5, color='black', fc='white'), True])
            min_time[rank] = t_start
            max_time[rank] = t_stop
          run_counter[rank] += 1
  with open(path) as file:
    for line in file:
      if line.strip().startswith("Model"):
        content = line.rstrip().replace(':', '|').split('|')
        op = content[8].strip()
        rank = int(content[2])
        t_start = float(content[4])
        t_stop = float(content[6])
        if op in colors and t_start > min_time[rank] and t_stop < max_time[rank]:
          rectangles.append(["", mpatch.Rectangle((float(content[4]), rank - .25), t_stop - t_start,
                                                  .5, color='black', fc=colors[op]), True])
          if op not in used_ops:
            used_ops[op] = None

  # Plot
  fig, ax = plt.subplots(1, 1, figsize=(8, 4.8))
  ax.title.set_text(type)
  leg = [
    Line2D([0], [0], marker='o', color='black', label=key, markerfacecolor=value, markersize=15, linestyle='None')
    for key, value in colors.items() if key in used_ops]
  leg += [Line2D([0], [0], marker='o', color='black', label='runBraid', markerfacecolor='white', markersize=15,
                 linestyle='None'),
          ]
  plt.legend(handles=leg, title='Task description', loc='upper center', bbox_to_anchor=(0.5, 1.17),
             ncol=5, fancybox=True, shadow=True, numpoints=1)

  for r in rectangles_r:
    rec = ax.add_artist(r[1])
  for r in rectangles:
    rec = ax.add_artist(r[1])
  ax.set_xlim(min(min_time), max(max_time))
  ax.set_yticks([x for x in range(procs)])
  ax.set_ylim(-.5, procs - .5)
  plt.show()

if __name__ == '__main__':
  parseOutputFile(path='path_to_file', run=1, procs=8)
