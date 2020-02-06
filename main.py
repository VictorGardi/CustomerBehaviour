import configparser as cp

config = cp.ConfigParser()

config.read('config.ini')

print(config['discrete_events_one_expert']['N_EXPERTS'])

