from gym.envs.registration import register

register(id='discrete-buying-events-v0', entry_point='custom_gym.envs:DiscreteBuyingEvents', kwargs={'sex':0, 'age':0, 'history':7*[0]})