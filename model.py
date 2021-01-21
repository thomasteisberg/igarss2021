import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import wandb


@tf.function
def tf_sign_not_zero(x):
    # A variant on the sign function where sign(0) = 1 but otherwise normal
    s = tf.math.sign(x)
    return tf.where(tf.math.equal(s,0), tf.ones_like(s), s)


@tf.function
def tf_error_outside_bounds(pred, obs, upper_bound, lower_bound):
    pos_error = tf.math.maximum(tf_sign_not_zero(obs)*pred - (tf.math.sign(obs)*obs + upper_bound), 0)
    neg_error = tf.math.minimum(tf_sign_not_zero(obs)*pred - (tf.math.sign(obs)*obs - tf.math.minimum(tf.cast(lower_bound, 'float'), tf.math.abs(obs))), 0)
    return tf.math.maximum(tf.square(pos_error), tf.square(neg_error))


class PINN(tf.keras.Model):
    def __init__(self, config, dataset, gen_model_filename=None):
        super().__init__()

        self.use_wandb = config.get('wandb', False)

        self.velocity_error_allowed = config.get('velocity_error_allowed', 0)
        self.allow_higher_depth_averaged_velocity = config.get('allow_higher_depth_averaged_velocity', True)
        if self.allow_higher_depth_averaged_velocity:
            self.velocity_error_upper = self.velocity_error_allowed
        else:
            self.velocity_error_upper = 0

        self.smoothing_norm_type = config.get('smoothing_norm_type', 2)
        self.radar_data_norm_type = config.get('radar_data_loss_norm_type', 2)

        self.gen_learning_rate = config.get('learning_rate', 0.001)

        self.is_1d = config.get('is_1d', False)

        self.predict_surface_velocity = config.get('predict_surface_velocity', False)

        self.loss_weights = {
            'radar_data': tf.Variable(config['radar_loss_weight'], dtype=tf.float32, trainable=False),
            'velocity_data': tf.Variable(config['velocity_loss_weight'], dtype=tf.float32, trainable=False),
            'negative_thickness': tf.Variable(config['negative_thickness_loss_weight'], dtype=tf.float32, trainable=False),
            'model': tf.Variable(config['model_loss_weight'], dtype=tf.float32, trainable=False),
            'thickness_smoothing': tf.Variable(config.get('thickness_smoothing_loss_weight',0), dtype=tf.float32, trainable=False),
            'velocity_smoothing': tf.Variable(config.get('velocity_smoothing_loss_weight',0), dtype=tf.float32, trainable=False),
            'velocity_diff_smoothing': tf.Variable(config.get('velocity_diff_smoothing_loss_weight',0), dtype=tf.float32, trainable=False),
            'velocity_mag_data': tf.Variable(config.get('velocity_mag_data_loss_weight',0), dtype=tf.float32, trainable=False),
            'velocity_ang_data': tf.Variable(config.get('velocity_ang_data_loss_weight',0), dtype=tf.float32, trainable=False),
            'surface_velocity_data': tf.Variable(config.get('surface_velocity_data_loss_weight',0), dtype=tf.float32, trainable=False)
        }

        self.dataset = dataset

        if gen_model_filename:
            self.make_loss_functions()
            gen_model = tf.keras.models.load_model(gen_model_filename,
                            custom_objects={
                                'radar_data_loss': self.radar_data_loss,
                                'velocity_data_loss': self.velocity_data_loss,
                                'model_loss': self.model_loss,
                                'negative_thickness_loss': self.negative_thickness_loss,
                                'gen_loss': self.gen_loss,
                                'unweighted_loss': self.unweighted_loss,
                                'thickness_smoothing_loss': self.thickness_smoothing_loss,
                                'velocity_smoothing_loss': self.velocity_smoothing_loss,
                                'velocity_diff_smoothing_loss': self.velocity_diff_smoothing_loss,
                                'velocity_mag_data_loss': self.velocity_mag_data_loss,
                                'velocity_ang_data_loss': self.velocity_ang_data_loss,
                                'surface_velocity_data_loss': self.surface_velocity_data_loss
                            })
            self.generator = gen_model
        else:
            self.generator = self.make_generator(config)
    
    def make_loss_functions(self):
        
        def radar_data_loss(obs, pred):
            h_pred = pred[:,0] * self.dataset.h_scale
            h = obs[:,0]

            finite_labels = tf.math.is_finite(h)

            if self.radar_data_norm_type == 1:
                data_loss = tf.reduce_mean(tf.abs(tf.boolean_mask(h, finite_labels) - tf.boolean_mask(h_pred, finite_labels)))
            else:
                data_loss = tf.reduce_mean(tf.square(tf.boolean_mask(h, finite_labels) - tf.boolean_mask(h_pred, finite_labels)))
            data_loss = tf.where(tf.math.is_nan(data_loss), tf.zeros_like(data_loss), data_loss)

            return data_loss
        self.radar_data_loss = radar_data_loss

        def velocity_data_loss(obs, pred):
            vx_pred = pred[:,1] * self.dataset.v_scale
            vy_pred = pred[:,2] * self.dataset.v_scale
            vx = obs[:,1]
            vy = obs[:,2]

            finite_labels = tf.math.is_finite(vx)

            vx_finite = tf.boolean_mask(vx, finite_labels)
            vy_finite = tf.boolean_mask(vy, finite_labels)
            vx_pred_finite = tf.boolean_mask(vx_pred, finite_labels)
            vy_pred_finite = tf.boolean_mask(vy_pred, finite_labels)

            vx_error = tf_error_outside_bounds(vx_pred_finite, vx_finite, self.velocity_error_upper, self.velocity_error_allowed)
            vy_error = tf_error_outside_bounds(vy_pred_finite, vy_finite, self.velocity_error_upper, self.velocity_error_allowed)

            velocity_data_loss = (tf.reduce_mean(vx_error) + tf.reduce_mean(vy_error)) / 2

            return velocity_data_loss
        self.velocity_data_loss = velocity_data_loss

        def surface_velocity_data_loss(obs, pred):
            surf_vx_pred = pred[:,3] * self.dataset.v_scale
            surf_vy_pred = pred[:,4] * self.dataset.v_scale
            vx = obs[:,1]
            vy = obs[:,2]

            finite_labels = tf.math.is_finite(vx)

            vx_diff = tf.boolean_mask(vx, finite_labels) - tf.boolean_mask(surf_vx_pred, finite_labels)
            vy_diff = tf.boolean_mask(vy, finite_labels) - tf.boolean_mask(surf_vy_pred, finite_labels)

            surf_velocity_data_loss = (tf.reduce_mean(tf.square(vx_diff)) + tf.reduce_mean(tf.square(vy_diff))) / 2

            return surf_velocity_data_loss
        self.surface_velocity_data_loss = surface_velocity_data_loss

        def velocity_mag_data_loss(obs, pred):
            vx_pred = pred[:,1] * self.dataset.v_scale
            vy_pred = pred[:,2] * self.dataset.v_scale
            vx = obs[:,1]
            vy = obs[:,2]

            finite_labels = tf.math.is_finite(vx)

            vx_finite = tf.boolean_mask(vx, finite_labels)
            vy_finite = tf.boolean_mask(vy, finite_labels)
            vx_pred_finite = tf.boolean_mask(vx_pred, finite_labels)
            vy_pred_finite = tf.boolean_mask(vy_pred, finite_labels)

            obs_mag = tf.math.sqrt(tf.square(vx_finite)+tf.square(vy_finite))
            obs_mag_safe = tf.where(tf.math.is_finite(obs_mag), obs_mag, tf.zeros_like(obs_mag))

            pred_mag = tf.math.sqrt(tf.square(vx_pred_finite)+tf.square(vy_pred_finite))
            pred_mag_safe = tf.where(tf.math.is_finite(pred_mag), pred_mag, tf.zeros_like(pred_mag))

            return tf.reduce_mean(tf_error_outside_bounds(pred_mag_safe, obs_mag_safe, self.velocity_error_upper, self.velocity_error_allowed))

        self.velocity_mag_data_loss = velocity_mag_data_loss

        def velocity_ang_data_loss(obs, pred):
            vx_pred = pred[:,1] * self.dataset.v_scale
            vy_pred = pred[:,2] * self.dataset.v_scale
            vx = obs[:,1]
            vy = obs[:,2]

            finite_labels = tf.math.is_finite(vx)

            vx_finite = tf.boolean_mask(vx, finite_labels)
            vy_finite = tf.boolean_mask(vy, finite_labels)
            vx_pred_finite = tf.boolean_mask(vx_pred, finite_labels)
            vy_pred_finite = tf.boolean_mask(vy_pred, finite_labels)

            theta_diff = tf.math.atan2(vy_finite, vx_finite) - tf.math.atan2(vy_pred_finite, vx_pred_finite)
            ang_loss = tf.reduce_mean(tf.square(theta_diff))

            return ang_loss
        self.velocity_ang_data_loss = velocity_ang_data_loss

        def model_loss(obs, predictions):
            x = obs[:,3] * self.dataset.xy_scale
            y = obs[:,4] * self.dataset.xy_scale
            
            # div(h*v) = h*vx_x + h_x*vx + h*vy_y + h_y*vy = 0

            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x)
                tape.watch(y)
                inpt = tf.stack([x / self.dataset.xy_scale, y / self.dataset.xy_scale], axis=1)
                pred2 = self.generator(inpt)
                
                h = pred2[:,0] * self.dataset.h_scale
                vx = pred2[:,1] * self.dataset.v_scale
                vy = pred2[:,2] * self.dataset.v_scale

            h_x = tape.gradient(h, x)
            h_y = tape.gradient(h, y)
            vx_x = tape.gradient(vx, x)
            vy_y = tape.gradient(vy, y)

            if self.is_1d:
                f = h*vx_x + h_x*vx
            else:
                f = h*vx_x + h_x*vx + h*vy_y + h_y*vy

            model_loss = tf.reduce_mean(tf.square(f))
            return model_loss
        self.model_loss = model_loss

        def thickness_smoothing_loss(obs, predictions):
            x = obs[:,3] * self.dataset.xy_scale
            y = obs[:,4] * self.dataset.xy_scale

            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x)
                tape.watch(y)
                inpt = tf.stack([x / self.dataset.xy_scale, y / self.dataset.xy_scale], axis=1)
                pred2 = self.generator(inpt)
                
                h = pred2[:,0] * self.dataset.h_scale

            h_x = tape.gradient(h, x) * 1000 # convert to m/km
            h_y = tape.gradient(h, y) * 1000

            if self.is_1d:
                h_y = 0 * h_y

            if self.smoothing_norm_type == 2:
                smoothing_loss = tf.reduce_mean(tf.square(h_x)) + tf.reduce_mean(tf.square(h_y))
            elif self.smoothing_norm_type == 1:
                smoothing_loss = tf.reduce_mean(tf.abs(h_x)) + tf.reduce_mean(tf.square(h_y))
            else:
                raise Exception(f"Unknown smoothing norm type {self.smoothing_norm_type} (try 1 or 2)")
            return smoothing_loss
        self.thickness_smoothing_loss = thickness_smoothing_loss


        def velocity_smoothing_loss(obs, predictions):
            x = obs[:,3] * self.dataset.xy_scale
            y = obs[:,4] * self.dataset.xy_scale

            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x)
                tape.watch(y)
                inpt = tf.stack([x / self.dataset.xy_scale, y / self.dataset.xy_scale], axis=1)
                pred2 = self.generator(inpt)
                
                vx = pred2[:,1] * self.dataset.v_scale
                vy = pred2[:,2] * self.dataset.v_scale

            vx_x = tape.gradient(vx, x) * 1000
            vx_y = tape.gradient(vx, y) * 1000
            vy_y = tape.gradient(vy, y) * 1000
            vy_x = tape.gradient(vy, x) * 1000

            if self.is_1d:
                vx_y = vx_y * 0
                vy_y = vy_y * 0
                vy_x = vy_x * 0

            
            smoothing_loss = (tf.reduce_mean(tf.square(vx_x)) + tf.reduce_mean(tf.square(vx_y)) + \
                tf.reduce_mean(tf.square(vy_y)) + tf.reduce_mean(tf.square(vy_x)))/4
            
            return smoothing_loss
        self.velocity_smoothing_loss = velocity_smoothing_loss

        def velocity_diff_smoothing_loss(obs, predictions):
            x = obs[:,3] * self.dataset.xy_scale
            y = obs[:,4] * self.dataset.xy_scale

            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x)
                tape.watch(y)
                inpt = tf.stack([x / self.dataset.xy_scale, y / self.dataset.xy_scale], axis=1)
                pred2 = self.generator(inpt)
                
                vx = pred2[:,1] * self.dataset.v_scale
                vy = pred2[:,2] * self.dataset.v_scale
                vx_surf = pred2[:,3] * self.dataset.v_scale
                vy_surf = pred2[:,4] * self.dataset.v_scale

                vx_diff = vx_surf - vx
                vy_diff = vy_surf - vy

            vx_diff_x = tape.gradient(vx_diff, x) * 1000 # per km
            vx_diff_y = tape.gradient(vx_diff, y) * 1000
            vy_diff_y = tape.gradient(vy_diff, y) * 1000
            vy_diff_x = tape.gradient(vy_diff, x) * 1000

            if self.is_1d:
                vy_diff_y = vy_diff_y * 0
                vy_diff_x = vy_diff_x * 0
                vx_diff_y = vx_diff_y * 0

            smoothing_loss = (tf.reduce_mean(tf.square(vx_diff_x)) + tf.reduce_mean(tf.square(vy_diff_y)) + \
                tf.reduce_mean(tf.square(vx_diff_y)) + tf.reduce_mean(tf.square(vy_diff_x)))/4

            return smoothing_loss
        self.velocity_diff_smoothing_loss = velocity_diff_smoothing_loss
        
        
        def negative_thickness_loss(obs, pred):
            h = pred[:,0]
            return tf.reduce_mean(tf.square(tf.minimum(h, 0)))
        self.negative_thickness_loss = negative_thickness_loss
        

        def gen_loss(obs, pred, weights=self.loss_weights):
            loss = weights.get('radar_data', 1.0) * radar_data_loss(obs, pred)
            loss += weights.get('velocity_data', 1.0) * velocity_data_loss(obs, pred)
            loss += weights.get('negative_thickness', 1.0) * negative_thickness_loss(obs, pred)
            loss += weights.get('model', 1.0) * model_loss(obs, pred)
            loss += weights.get('thickness_smoothing', 1.0) * thickness_smoothing_loss(obs, pred)
            loss += weights.get('velocity_smoothing', 1.0) * velocity_smoothing_loss(obs, pred)
            
            loss += weights.get('velocity_mag_data', 1.0) * velocity_mag_data_loss(obs, pred)
            loss += weights.get('velocity_ang_data', 1.0) * velocity_ang_data_loss(obs, pred)
            
            if self.predict_surface_velocity:
                loss += weights.get('velocity_diff_smoothing', 1.0) * velocity_diff_smoothing_loss(obs, pred)
                loss += weights.get('surface_velocity_data', 1.0) * surface_velocity_data_loss(obs, pred)
            return loss
        self.gen_loss = gen_loss


        def unweighted_loss(obs, pred):
            return gen_loss(obs, pred, weights={})
        self.unweighted_loss = unweighted_loss



    def compile(self):
        super(PINN, self).compile()

        self.make_loss_functions()

        gen_opt = tf.keras.optimizers.Adam(learning_rate=self.gen_learning_rate)

        metrics = [self.unweighted_loss, self.radar_data_loss, self.velocity_data_loss, self.model_loss,
                    self.negative_thickness_loss, self.thickness_smoothing_loss, self.velocity_smoothing_loss,
                    self.velocity_mag_data_loss, self.velocity_ang_data_loss]
        if self.predict_surface_velocity:
            metrics.append(self.velocity_diff_smoothing_loss)
            metrics.append(self.surface_velocity_data_loss)

        self.generator.compile(optimizer=gen_opt, loss=self.gen_loss, metrics=metrics)
    

    @property
    def metrics(self):
        return self.generator.metrics


    def call(self, inputs):
        return self.generator(inputs)

    
    def make_generator(self, config):
        if self.predict_surface_velocity:
            n_outputs = 5
        else:
            n_outputs = 3

        layers = [2] + [config['generator_width']]*(config['generator_layers']) + [n_outputs]

        model = tf.keras.Sequential(name="generator")
        model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))

        for width in layers[1:-1]:
            model.add(tf.keras.layers.Dense(
                width, activation=config['generator_activation'],
                kernel_initializer="glorot_normal"))
        model.add(tf.keras.layers.Dense(
                layers[-1], activation='linear',
                kernel_initializer="glorot_normal"))
    
        return model
    

    def predict_from_unnormalized(self, x, y):
        x_norm, y_norm, _, _, _ = self.dataset.normalize(x, y, None, None, None)
        inpts = np.concatenate([np.expand_dims(x_norm, -1), np.expand_dims(y_norm, -1)], axis=1)
        pred = self.generator(inpts)
        return {
            'h': pred[:,0] * self.dataset.h_scale,
            'vx': pred[:,1] * self.dataset.v_scale,
            'vy': pred[:,2] * self.dataset.v_scale
        }
