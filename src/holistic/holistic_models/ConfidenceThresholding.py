from interface.WTA import WTA

class ConfidenceThresholdingWtaModule(WTA):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def wta(self, t, prob, nav_outs):
        probs, a_t = prob.max(1)  
        probs_cpu = probs.cpu()
        return [prob < self.threshold for prob in probs_cpu]


class CappedConfidenceWtaModule(WTA):
    """Ask when navigation confidence is below the threshold, but at most
    ``cap`` questions per episode. The cap keeps the dialog turn count close
    to the human dialog length, which the per-sample score rewards."""

    def __init__(self, threshold=0.6, cap=2, min_step=0):
        self.threshold = threshold
        self.cap = cap
        self.min_step = min_step
        self.ask_counts = []

    def wta(self, t, prob, nav_outs):
        probs, _ = prob.max(1)
        probs_cpu = probs.cpu()
        n = len(probs_cpu)
        if t == 0:
            self.ask_counts = [0] * n
        ask = []
        for i, p in enumerate(probs_cpu):
            if t >= self.min_step and p < self.threshold and self.ask_counts[i] < self.cap:
                self.ask_counts[i] += 1
                ask.append(True)
            else:
                ask.append(False)
        return ask
