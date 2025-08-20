import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_combined_csvs(csv_files, line1_value=None, line2_value=None):
    plt.figure(figsize=(10, 6))

    # Diccionario de etiquetas personalizadas
    custom_labels = {
        'GPT_SAC_4M.csv': 'GPT_SAC',
        'GPT_PPO_4M.csv': 'GPT_PPO',
        'DEEPSEEK_PPO_4M.csv': 'DEEPSEEK_PPO',
        'DEEPSEEK_SAC_4M.csv': 'DEEPSEEK_SAC',
    }

    # Separar archivos por grupo (DEEPSEEK vs GPT)
    deepseek_files = [f for f in csv_files if 'DEEPSEEK' in f.upper()]
    gpt_files = [f for f in csv_files if 'GPT' in f.upper()]

    # Paletas de colores
    deepseek_colors = sns.color_palette("Blues", len(deepseek_files))
    gpt_colors = sns.color_palette("Reds", len(gpt_files))

    # --- Plot DEEPSEEK ---
    for idx, csv_file in enumerate(deepseek_files):
        df = pd.read_csv(csv_file)
        if all(col in df.columns for col in ['Step', 'Value']):
            label = custom_labels.get(os.path.basename(csv_file), os.path.basename(csv_file))
            sns.lineplot(x='Step', y='Value', data=df, label=label, color=deepseek_colors[idx])
        else:
            print(f"❌ Columnas faltantes en: {csv_file}")

    # --- Plot GPT ---
    for idx, csv_file in enumerate(gpt_files):
        df = pd.read_csv(csv_file)
        if all(col in df.columns for col in ['Step', 'Value']):
            label = custom_labels.get(os.path.basename(csv_file), os.path.basename(csv_file))
            sns.lineplot(x='Step', y='Value', data=df, label=label, color=gpt_colors[idx])
        else:
            print(f"❌ Columnas faltantes en: {csv_file}")

    # Líneas horizontales de referencia
    if line1_value is not None:
        plt.axhline(y=line1_value, color='red', linestyle='--', label='Seals Expert')
    if line2_value is not None:
        plt.axhline(y=line2_value, color='blue', linestyle='--', label='FineTuned Expert')

    plt.title('Hopper Mean Reward')
    plt.xlim(left=0)   # solo límite inferior, ajusta según tus steps
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Lista de archivos CSV a combinar
csv_files = [
    'T2R_HOPPER_COMPARE/RewardsGraphs/DEEPSEEK_PPO_4M.csv',
    'T2R_HOPPER_COMPARE/RewardsGraphs/DEEPSEEK_SAC_4M.csv',
    'T2R_HOPPER_COMPARE/RewardsGraphs/GPT_PPO_4M.csv',
    'T2R_HOPPER_COMPARE/RewardsGraphs/GPT_SAC_4M.csv'
]

plot_combined_csvs(csv_files)
