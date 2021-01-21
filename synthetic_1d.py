import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import tensorflow as tf

matplotlib.use('svg')
new_rc_params = {
    "font.family": 'Times',
    "font.size": 12,
    "font.serif": [],
    "svg.fonttype": 'none'}
matplotlib.rcParams.update(new_rc_params)

np.random.seed(1)

n_obs_pts = 40

# Setup geometry
xs = np.linspace(0,20000,50)

v_true = 100 + 10*np.exp((xs-1000) / 7000) + 20*np.sin(xs/2000)#+ np.cumsum(np.random.normal(scale=1, size=np.shape(xs)))
flux_true = 100000
h_true = (flux_true / v_true)

observed_idx = np.concatenate([
                np.random.randint(0, int(len(xs)/3), size=(int(n_obs_pts*0.5),)),
                np.random.randint(int(2*len(xs)/3), int(len(xs)), size=(int(n_obs_pts*0.5),))],axis=0)
h_obs = h_true[observed_idx]
x_obs = xs[observed_idx]

v_obs = v_true


fig, axs = plt.subplots(3,1, sharex=True)
axs[0].plot(xs/1000, v_true)
axs[0].scatter(xs/1000, v_obs, label='Measurements', s=5)
axs[0].set_ylabel('velocity [m/yr]')
axs[0].legend()
axs[1].plot(xs/1000, h_true)
axs[1].scatter(x_obs/1000, h_obs, label='Measurements', s=5)
axs[1].set_ylabel('thickness [m]')
axs[1].legend()
axs[2].plot(xs/1000, h_true/1000*v_true/1000)
axs[2].set_ylabel('flux [$\mathregular{km^2}$ /yr]')
axs[2].set_xlabel('Distance along flowline [km]')

fig.savefig('1d-geometry.svg', format='svg', dpi=1000)

with open('synthetic-1d.pickle', 'wb') as f:
    pickle.dump({
        'radar': {
            'x': x_obs,
            'y': np.zeros(np.shape(x_obs)),
            'h': h_obs
        },
        'velocity': {
            'x': xs,
            'y': np.zeros(np.shape(xs)),
            'vx': v_obs,
            'vy': np.zeros(np.shape(xs))
        },
        'truth': {
            'x': xs,
            'h': h_true,
            'v': v_true
        }
    }, f)
