from functools import partial

import tensorflow as tf
import numpy as np

class PINNDataset(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size=32, n_random=1000, verbose=False, mode='pigan'):

        self.rng = np.random.default_rng()

        self.batch_size = batch_size
        self.verbose = verbose
        self.mode = mode

        # Store velocity data
        self.v = {}
        self.v['vx'] = data['velocity']['vx'].flatten()
        self.v['x'] = data['velocity']['x'].flatten()
        self.v['vy'] = data['velocity']['vy'].flatten()
        self.v['y'] = data['velocity']['y'].flatten()

        self.v_scale = np.mean([np.std(self.v['vx']), np.std(self.v['vy'])])*3

        # Store radar data
        self.r = {}
        self.r['h'] = data['radar']['h'].flatten()
        self.r['x'] = data['radar']['x']
        self.r['y'] = data['radar']['y']
        if 'v_nn' in data['radar']:
            self.r['v_nn'] = data['radar']['v_nn']

        self.h_scale = np.std(self.r['h'])*3

        all_x = np.concatenate([self.r['x'], self.v['x']])
        all_y = np.concatenate([self.r['y'], self.v['y']])
        self.x_center = np.mean(all_x)
        self.y_center = np.mean(all_y)
        self.xy_scale = np.mean([np.std(all_x), np.std(all_y)])

        # Generator training data

        r_x_norm, r_y_norm, _, _, _ = self.normalize(self.r['x'], self.r['y'], None, None, None)
        v_x_norm, v_y_norm, _, _, _ = self.normalize(self.v['x'], self.v['y'], None, None, None)

        random_x_norm = self.rng.uniform(np.min(v_x_norm), np.max(v_x_norm), size=(n_random,))
        random_y_norm = self.rng.uniform(np.min(v_y_norm), np.max(v_y_norm), size=(n_random,))

        len_radar = len(r_x_norm)
        len_vel = len(v_x_norm)

        self.inpts = np.concatenate([np.expand_dims(np.concatenate([r_x_norm, v_x_norm, random_x_norm]), -1),
                                     np.expand_dims(np.concatenate([r_y_norm, v_y_norm, random_y_norm]), -1)], axis=1)
        self.obs   = np.concatenate([np.expand_dims(np.concatenate([self.r['h'], np.nan * np.zeros((len_vel+n_random,))]), -1),
                                     np.expand_dims(np.concatenate([np.nan * np.zeros((len_radar,)), self.v['vx'], np.nan * np.zeros((n_random,))]), -1),
                                     np.expand_dims(np.concatenate([np.nan * np.zeros((len_radar,)), self.v['vy'], np.nan * np.zeros((n_random,))]), -1),
                                     np.expand_dims(np.concatenate([r_x_norm, v_x_norm, random_x_norm]), -1),
                                     np.expand_dims(np.concatenate([r_y_norm, v_y_norm, random_y_norm]), -1)], axis=1)

        ordering = np.arange(0, np.shape(self.inpts)[0])
        np.random.shuffle(ordering)
        self.inpts = self.inpts[ordering,:]
        self.obs = self.obs[ordering,:]

        # Setup batches

        self.n_batches = int(np.ceil((len_radar + len_vel) / self.batch_size))
        self.on_epoch_end()


    def normalize(self, x, y, vx, vy, h):
        if x is not None:
            norm_x = (x - self.x_center) / self.xy_scale
        else:
            norm_x = None
        
        if y is not None:
            norm_y = (y - self.y_center) / self.xy_scale
        else:
            norm_y = None
        
        if vx is not None:
            norm_vx = vx / self.v_scale
        else:
            norm_vx = None

        if vy is not None:
            norm_vy = vy / self.v_scale
        else:
            norm_vy = None
        
        if h is not None:
            norm_h = h / self.h_scale
        else:
            norm_h = None
        
        return norm_x, norm_y, norm_vx, norm_vy, norm_h


    def unnormalize(self, x, y, vx, vy, h):
        if x is not None:
            norm_x = (x*self.xy_scale) + self.x_center
        else:
            norm_x = None
        
        if y is not None:
            norm_y = (y*self.xy_scale) + self.y_center
        else:
            norm_y = None
        
        if vx is not None:
            norm_vx = vx * self.v_scale
        else:
            norm_vx = None

        if vy is not None:
            norm_vy = vy * self.v_scale
        else:
            norm_vy = None
        
        if h is not None:
            norm_h = h * self.h_scale
        else:
            norm_h = None
        
        return norm_x, norm_y, norm_vx, norm_vy, norm_h

    # Interface methods for Sequence

    def __len__(self):
        return self.n_batches

    def __getitem__(self, idx):
        # Generator
        first_idx, last_idx = idx*self.batch_size, np.minimum((idx+1)*self.batch_size, np.shape(self.inpts)[0])
        gen_x = self.inpts[first_idx:last_idx, :]
        gen_y = self.obs[first_idx:last_idx, :]

        
        return gen_x, gen_y
