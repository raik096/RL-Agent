


Python
markdown_content = """# Progetto GDL 25/26: Proximal Policy Optimization (PPO)
### Corso di Deep Learning - Università di Pisa (UNIPI)

Questo documento riassume in modo sintetico e strutturato il funzionamento e l'architettura dell'algoritmo **PPO (Proximal Policy Optimization)** con **Backbone Condiviso**, implementato per il progetto d'esame.

---

## 🗺️ Architettura del Sistema & Flusso dei Dati

Di seguito viene mostrato lo schema logico dell'agente. Il cuore del sistema risiede nel **Backbone Condiviso**, che elabora lo stato dell'ambiente una sola volta, estraendo le feature utili sia all'Attore che al Critico, ottimizzando drasticamente le prestazioni computazionali.



Code output
File creato con successo!

```text
                        +-------------------+
                        |   Stato (State)   |
                        +---------+---------+
                                  |
                                  v
                        +-------------------+
                        |  BACKBONE COMUNE  |
                        | (Feature Extr.)   |
                        +----+---------+----+
                             |         |
              +--------------+         +--------------+
              |                                       |
              v                                       v
      +---------------+                       +---------------+
      |  ACTOR HEAD   |                       |  CRITIC HEAD  |
      | (Policy/Azione)                       | (Value/Stato) |
      +-------+-------+                       +-------+-------+
              |                                       |
              v [Softmax]                             v [Linear]
      Distribuzione Proba.                    Valore dello Stato
        Categorica P(a|s)                            V(s)
              |                                       |
              v                                       |
       Scelta Azione A_t                              |
              |                                       |
              +----------------+----------------------+
                               |
                               v
                      +-----------------+
                      |    PPOMemory    |
                      | (Replay Buffer) |
                      +-----------------+
                      | Memorizza fino  |
                      | a T = 20 passi  |
                      +--------+--------+
                               |
                               v (Ogni 4 Epoche)
                     [ CAMPIONAMENTO BATCH ]
                               |
                               v
                     [ CALCOLO LOSS TOTALE ]
                               |
                               v
                     [ AGGIORNAMENTO PESI ]


📝 Sintesi del Procedimento ed Equazioni
L'addestramento si basa su una traiettoria limitata a $T = 20$ passi memorizzata nel buffer, dopodiché l'esperienza viene mescolata e suddivisa in mini-batch per aggiornare la rete tramite la Loss Totale.
1. Il Reparto Memoria (PPOMemory)
Raccoglie in modo sequenziale: Stati, Azioni, Ricompense, Valori V(s), Log-Probability, e flag Done.
Campionamento: Allo scadere dei $T$ passi, la funzione generate_batches() genera indici casuali (tramite np.arange e np.random.shuffle) rompendo la correlazione temporale ed evitando il sovraddestramento su stati consecutivi.
2. Il Critico (CRITIC)
Scopo: Valutare la bontà dello stato corrente stimando il valore atteso dei ritorni futuri ($V(s)$).
Aggiornamento: Vuole rendere la sua stima il più vicina possibile al ritorno effettivo ottenuto dall'esperienza.
Loss: Errore Quadratico Medio (MSE) tra il ritorno reale (Vantaggio + Valore stimato precedente) e la nuova previsione della rete.
$$\mathcal{L}_{critic} = \text{MSE}(\text{Ritorno} - V_{\theta}(s_t))$$
3. L'Attore (ACTOR)
Scopo: Decidere l'azione migliore da intraprendere calcolando una distribuzione di probabilità sulle azioni disponibili (gestita tramite una distribuzione Categorical e attivazione Softmax).
Clipping della Loss: Per evitare che la policy cambi troppo drasticamente tra un aggiornamento e l'altro (rompendo l'apprendimento a causa di probabilità che finiscono al denominatore), il PPO "clippa" il rapporto di probabilità $r_t(\theta)$ all'interno di un intervallo sicuro $[1-\epsilon, 1+\epsilon]$.
Il Vantaggio ($A_t$): Sfrutta il contributo del critico per capire se l'azione ha performato meglio o peggio del previsto, abbassando notevolmente la varianza rispetto ai metodi Policy Gradient classici.
4. La Loss Totale Co-Ottimizzata
Avendo un backbone condiviso, l'agente ottimizza una singola Loss combinata che bilancia i diversi scopi delle teste:

$$\text{LOSS}_{totale}(\theta) = \mathbb{E}_t \left[ \mathcal{L}_{actor}(\theta) - c_1 \cdot \mathcal{L}_{critic}(\theta) + c_2 \cdot \mathcal{S}_{entropy}[\pi_{\theta}](s_t) \right]$$
$\mathcal{L}_{actor}$ (Segno Positivo): Massimizza la probabilità delle azioni che generano un vantaggio positivo.
$c_1 \cdot \mathcal{L}_{critic}$ (Sottratta): Funziona da penalità; spinge l'ottimizzatore a minimizzare l'errore del Critico. Il coefficiente $c_1$ scherma il backbone comune da variazioni troppo aggressive guidate solo dal valore del Critico.
$c_2 \cdot \mathcal{S}_{entropy}$ (Sommata): Introduce un premio di entropia per mantenere la distribuzione leggermente uniforme. Impedisce alla rete di collassare prematuramente su un'unica mossa funzionante, incentivando l'esplorazione di nuove strategie.
Nota Implementativa: I mini-batch estratti vengono processati dall'ottimizzatore optim.Adam ripetendo l'aggiornamento per 4 epoche prima di ripulire interamente la memoria (clear_memory()) e ricominciare un nuovo ciclo di interazione on-policy.
"""
with open("Procedimento_PPO_UNIPI.md", "w", encoding="utf-8") as f:
f.write(markdown_content)
print("File creato con successo!")



Your Markdown file is ready
[file-tag: code-generated-file-0-1782487972097700631]

Ecco pronto il file Markdown! 

Ho creato un diagramma in ASCII art pulito e professionale che mostra la struttura a backbone condiviso con lo sdoppiamento tra Actor e Critic, unito all'interazione con il Replay Buffer (`PPOMemory`). 

Sotto il disegno ho riassunto i concetti chiave che hai scritto nei tuoi appunti, formattando le formule matematiche e spiegando i vari coefficienti e l'uso dell'entropia. È ottimo da usare come file `README.md` da caricare su GitHub o da allegare al tuo progetto universitario!


