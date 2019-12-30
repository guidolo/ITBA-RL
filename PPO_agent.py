
import numpy as np

from base_agent import BaseAgent

import keras.backend as K
from keras.layers import Dense, Input
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD

class ReinforceAgent(BaseAgent):
    
    def proximal_policy_optimization_loss(self, advantage, old_prediction):
        def loss(y_true, y_pred):
            prob = y_true * y_pred
            old_prob = y_true * old_prediction
            r = prob/(old_prob + 1e-10)
            return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - self.LOSS_CLIPPING, max_value=1 + self.LOSS_CLIPPING) * advantage) 
                           + self.ENTROPY_LOSS * -(prob * K.log(prob + 1e-10)))
        return loss

    def get_policy_model(self, lr=0.001, hidden_layer_neurons = 128, input_shape=[4], output_shape=2):
        
        ## Defino métrica - loss sin el retorno multiplicando
        def actor_loss(y_true, y_pred):
            prob = y_true * y_pred
            old_prob = y_true * old_prediction
            r = prob/(old_prob + 1e-10)
            return K.max(r)
        
        state_input = Input(shape=input_shape)
        advantage = Input(shape=(1,))
        old_prediction = Input(shape=(output_shape,))

        x = Dense(hidden_layer_neurons, activation='relu')(state_input)
        out_actions = Dense(output_shape, activation='softmax', name='output')(x)
        model_train = Model(inputs=[state_input, advantage, old_prediction], outputs=[out_actions])
        model_predict = Model(inputs=[state_input], outputs=[out_actions])
        
        model_train.compile(Adam(lr), loss=[self.proximal_policy_optimization_loss(advantage, old_prediction)], metrics=[actor_loss])
        return model_train, model_predict
    
    def get_action(self, eval=False):
        obs = self.scaler.transform(self.observation.reshape(1, self.nS))
        obs = self.observation.reshape(1, self.nS)
        p = self.model_predict.predict(obs)
        if eval is False:
            action = np.random.choice(self.nA, p=p[0]) #np.nan_to_num(p[0])
        else:
            action = np.argmax(p[0])
        action_one_hot = np.zeros(self.nA)
        action_one_hot[action] = 1
        return action, action_one_hot, p
    
    def get_entropy(self, preds, epsilon=1e-12):
        entropy = np.mean(-np.sum(np.log(preds+epsilon)*preds, axis=1)/np.log(self.nA))
        return entropy
    
    def get_critic_model(self, lr=0.001, hidden_layer_neurons = 128, input_shape=[4], output_shape=1):
        model = Sequential()
        model.add(Dense(hidden_layer_neurons, input_shape=input_shape, activation='relu'))
#         model.add(Dense(hidden_layer_neurons, input_shape=input_shape, activation='selu'))
        model.add(Dense(output_shape, activation='linear'))
        model.compile(Adam(lr), loss=['mse'])
        return model
    
    def get_discounted_rewards(self, r):
        # Por si es una lista
        r = np.array(r, dtype=float)
        """Take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r 