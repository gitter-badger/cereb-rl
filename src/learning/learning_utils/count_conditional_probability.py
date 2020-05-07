class CountConditionalProbability:
    """
    A simple class to estimate probabilities based on counts
    """

    def __init__(self):

        self._total_count = 0.0
        self._values = {}

    def add(self, new_entry):

        self._total_count += 1

        if new_entry in self._values:
            self._values[new_entry] += 1.0
        else:
            self._values[new_entry] = 1.0

    def get_probability(self):

        z = float(max(1.0, self._total_count))
        prob = [(key, value / z) for (key, value) in self._values.items()]

        return prob