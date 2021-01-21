import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

import pickle

import tensorflow as tf

import wandb

class GeneratorVisualizationCallback(tf.keras.callbacks.Callback):
    def __init__(self, config, dataset, pigan, every_n_epochs=1):
        super().__init__()

        self.config = config
        self.dataset = dataset
        self.pigan = pigan

        self.every_n_epochs = every_n_epochs

    def roi_prediction(self, roi):
        if roi.get('1d', False):
            xs = np.arange(roi['x0'], roi['x1'], roi['spacing'])
            return xs, None, self.pigan.predict_from_unnormalized(xs, np.zeros(np.shape(xs)))
        else:
            xs = np.arange(roi['x0'], roi['x1'], roi['spacing'])
            ys = np.arange(roi['y0'], roi['y1'], roi['spacing'])
            X, Y = np.meshgrid(xs, ys)

            return X, Y, self.pigan.predict_from_unnormalized(X.flatten(), Y.flatten())

    def plot_prediction(self, X, Y, data, norm=None, title=""):
        fig, ax = plt.subplots(figsize=(5,5))

        if Y is not None: # 2D
            if norm is not None:
                sc = ax.scatter(X.flatten(), Y.flatten(), c=data, s=0.3, norm=norm)
                fig.colorbar(sc, ax=ax)
            else:
                sc = ax.scatter(X.flatten(), Y.flatten(), c=data, s=0.3)
                fig.colorbar(sc, ax=ax)

            ax.set_aspect('equal')
        else: # 1D
            ax.scatter(X.flatten(), data)

        ax.set_title(title)

        return fig, ax
    
    def on_epoch_end(self, epoch, logs=None):
        if not (epoch % self.every_n_epochs == 0):
            return

        for roi in self.config['eval_regions']:
            X, Y, pred = self.roi_prediction(roi)

            h_norm = matplotlib.colors.Normalize(vmin=0,
                                                vmax=np.max(self.dataset.r['h'])+250)

            if 'h' in pred:
                fig, ax = self.plot_prediction(X, Y, pred['h'], title=f"{roi['title']} at epoch {epoch}", norm=h_norm)
                wandb.log({f"thickness_{roi['title']}": wandb.Image(fig)})

            if 'vx' in pred:
                fig, ax = self.plot_prediction(X, Y, pred['vx'], title=f"vx: {roi['title']} at epoch {epoch}")
                wandb.log({f"vx_{roi['title']}": wandb.Image(fig)})

                fig, ax = self.plot_prediction(X, Y, pred['vy'], title=f"vy: {roi['title']} at epoch {epoch}")
                wandb.log({f"vy_{roi['title']}": wandb.Image(fig)})
            
                fig, ax = self.plot_prediction(X, Y, np.sqrt(pred['vx']**2 + pred['vy']**2), title=f"v: {roi['title']} at epoch {epoch}")
                wandb.log({f"v_{roi['title']}": wandb.Image(fig)})

            plt.close('all')

