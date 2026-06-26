"""
PROXIMAL POLICY OPTIMIZATION (PPO) PER PROGETTO UNIPI GDL 25/26: 

Abbiamo un backbone che viene condiviso da ACTOR e CRITIC, queste due reti
hanno un ultimo layer separato con pesi diversi ma tutta la parte di estra
_zione di feature, la condividono. In questo modo si allena una sola parte
una sola volta. Hanno però ovviamente scopi diversi:

----------------------------------------------------------------
-------------------------- CRITIC ------------------------------
                        valuta lo stato

                        
        Come si aggiorna:

            Più semplice dell'ACTOR, il suo obiettivo è imparare a stimare il valore degli s
            _tati quindi vuole minimizzare l'errore tra le sue previsioni e i risultati effe
            _ttivi. Abbiamo bisogno del: 
            ritorno = advantage + value del critico (dalla memoria) e
            L_critic = MSE(ritorno - value del critico(dalla rete))
             
----------------------------------------------------------------
--------------------------- ACTOR ------------------------------
decide cosa fare basandosi sullo stato corrente in base alla policy


        Come si aggiorna:

            La policy che è una distrubuzione dell'azione condizionata allo stato in cui è, 
            potrebbe rovinare e rompere l'apprendimento, infatti essendo prima posto al den
            _ominatore una probabilità è possibile che il numero scoppi scostando e mettendo
            sul gradiente un numero troppo grande da gestire, quindi si è proceduto a clippa
            _re questo valore in modo da renderlo sempre piccolo e sicuro, ma correggendolo 
            fortemente se c'è un grave peggioramento delle prestazioni. Questo accade con il 
            clipping della loss. A_t (dove t passi è molto minore dei passi totali T perché 
            non è conveniente nel momento che per aggiornare dovrebbe fare tutta la partita
            oltre che a diminure la varianza) è il VANTAGGIO che viene calcolato su tutti 
            gli stati, quelli reali (quindi quelli che l'agente ha effettivamente fatto), qu
            _elli di ora e quelli futuri. Ed è ottimo perché almeno l'agente si ferma, e tira
            le somme grazie anche al contributo del critico che prevede e che abbassa la vari
            _anza, poi c'è lo sconto gamma e si sottrae V(s_t) che è la stima del critico in
            base allo stato di partenza

----------------------------------------------------------------
----------------------------------------------------------------

La LOSS totale prevede l'insieme delle due LOSS da impostare per massimizzare il gradiente:

    LOSS_totale(theta) = 
        VAL_ATTESO[L_actor(theta) - c1 * L_critic(theta) + c2 * S[policy_theta](s_t)]

    note:
        1)  L_actor ha segno positivo perché l'obiettivo dell'attore è aumentare la probabilità
            delle azioni che hanno portato ad un vantaggio.
        2)  L_critic è negativo, infatti lo scopo è che va minimizzato, sottraendolo in un sistema
            spinge verso l'alto porta a trovare valori che diminuiscano questa sottrazione riducen
            _dolo al minimo
        3)  Il coefficiente c1 serve a rendere l'aggiornamento dell'errore meno aggressivo e l'ent
            _ropia ci permette invece di aggiungere dell'incertezza sulla distribuzione rendendola
            più uniforme e quindi meno prevedibile, mitiga la rete a rendere troppo probabile una 
            mossa perché funzionante, in modo da esplorare altre soluzioni


La rete restituisce delle probabilità per una distribuzione

---- DETTAGLI IMPLEMENTATIVI:

ReplayBuffer: batch[];
ActorNetwork: Class
CriticNetwork: Class
Agent: Class
Il main farà il loop per il train e la valutazione

La memoria verrà bloccata a T passi, diciamo 20, e terremo traccia degli:
    - stati
    - azioni
    - ricompense
    - fatti
    - valori
    - log probs

I batches vengono pescati randomicamente da un bacino di batch e verranno 
aggiornati ogni 4 epoche
"""

import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.actions = []
        self.vals = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size


    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(n_states, dtype=np.int64)
        indices = np.range(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches
    
    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.probs.append(probs)
        self.vals.append(vals)
        self.action.append(action)
        self.reward.append(reward)
        self.done.append(done)

    def clear_memory(self):
        self.states = []
        self.probs =[]
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []



print("Hello!"); 


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input, input_dims, alpha,
                 fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        if T.cuda.is_available():
            self.device = T.device('cuda')
        elif T.backends.mps.is_available():
            self.device = T.device('mps')
        else:
            self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)

        return dist
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    
class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256,
                 chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1) # Un solo valore di output
        )

        """
        Il Learning Rate alpha è lo stesso per entrambe le reti, generalmente viene dato
        un learning rate più alto a al Critic e uno più basso all'Actor, trattandosi di un
        problema di regressione, vogliamo che il Critic impari velocemente a mappare gli stati
        i loro valori reali, L'Actor invece preferisce essere più conservativo essendo la policy
        un tradeoff delicato.
        """
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        if T.cuda.is_available():
            self.device = T.device('cuda')
        elif T.backends.mps.is_available():
            self.device = T.device('mps')
        else:
            self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
