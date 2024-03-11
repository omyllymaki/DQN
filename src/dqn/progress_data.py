from src.dqn.transition import Transition


class ProgressData:

    def __init__(self):
        self.transitions = []
        self.losses = []
        self.temporal_diff_errors = []

    def push_transition(self, state, action, next_state, reward) -> None:
        transition = Transition(state=state,
                                action=action,
                                next_state=next_state,
                                reward=reward,
                                priority=None)
        self.transitions.append(transition)

    def push_losses(self, losses):
        self.losses.append(losses)

    def push_temporal_difference_errors(self, temporal_diff_errors):
        self.temporal_diff_errors.append(temporal_diff_errors)
