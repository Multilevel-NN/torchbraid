

class BeamHypo:
  def __init__(self, num_beams, len_penalty):
    self.num_beams = num_beams
    self.len_penalty = len_penalty

    self.beams = []
    self.worst_score = 1e9

  def add(self, hypo, sum_logprobs):
    score = sum_logprobs / len(hypo)**self.len_penalty

    if len(self.beams) < self.num_beams or score > self.worst_score:
      self.beams.append((score, hypo))
      self.beams.sort(key=lambda x: x[0],#[(score, -idx) for idx, (score, hypo) in enumerate(self.beams)], 
        reverse=True,
      )

      if len(self.beams) > self.num_beams:
        _ = self.beams.pop()

      self.worst_score = self.beams[-1][0]

  def is_done(self, best_sum_logprobs, curr_len):
    if len(self.beams) < self.num_beams:
      return False

    new_score = best_sum_logprobs / curr_len**self.len_penalty
    return new_score <= self.worst_score





































