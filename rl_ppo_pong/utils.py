import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file, window=100, target_score=None):
    """
    Traccia la curva di apprendimento con punteggi grezzi, media mobile e deviazione standard.
    Ottimizzato per report accademici.
    """
    scores = np.array(scores)
    running_avg = np.zeros(len(scores))
    std_dev = np.zeros(len(scores))

    # Calcolo della media mobile e della deviazione standard
    for i in range(len(running_avg)):
        window_slice = scores[max(0, i - window + 1):(i + 1)]
        running_avg[i] = np.mean(window_slice)
        std_dev[i] = np.std(window_slice)

    # Impostazione della figura (dimensioni generose per i report)
    plt.figure(figsize=(10, 6))

    # 1. I punteggi grezzi (in sottofondo, semi-trasparenti)
    plt.plot(x, scores, alpha=0.2, color='gray', label='Punteggio Grezzo')

    # 2. L'ombra della deviazione standard (mostra la stabilità)
    plt.fill_between(x, running_avg - std_dev, running_avg + std_dev, 
                     color='#1f77b4', alpha=0.2, label='Deviazione Standard')

    # 3. La media mobile (in primo piano, marcata)
    plt.plot(x, running_avg, color='#1f77b4', linewidth=2, label=f'Media Mobile ({window} ep.)')

    # 4. (Opzionale) Linea orizzontale per il punteggio target/massimo
    if target_score is not None:
        plt.axhline(y=target_score, color='red', linestyle='--', alpha=0.8, 
                    label=f'Target ({target_score})')

    # Estetica Magistrale
    plt.title('Curva di Apprendimento dell\'Agente PPO', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Episodi', fontsize=12, fontweight='bold')
    plt.ylabel('Punteggio', fontsize=12, fontweight='bold')
    
    # Griglia e Legenda
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best', fontsize=10)
    
    # Rimuove gli spazi bianchi inutili ai bordi
    plt.tight_layout()

    # Salvataggio in alta risoluzione (300 dpi è lo standard per la stampa/PDF)
    plt.savefig(figure_file, dpi=300, bbox_inches='tight')
    
    # Chiude la figura per liberare la memoria (fondamentale se chiamato più volte nel loop)
    plt.close()