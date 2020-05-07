class StationaryActionConditionPolicy:
    """ A policy that takes action by evaluating an input condition """

    def __init__(self, action_condition):
        self.action_condition = action_condition

    def gen_q_val(self, observations):
        raise NotImplementedError()

    def sample_action(self, observations):
        return self.action_condition(observations)

    def get_argmax_action(self, observations):
        return self.action_condition(observations)

    def save(self, folder_name, model_name=None):
        raise NotImplementedError()

    def load(self, folder_name, model_name=None):
        raise NotImplementedError()