from collections import namedtuple

from recordclass import recordclass

Transition = recordclass('Transition',
                         ('state', 'action', 'next_state', 'reward', 'priority'))
