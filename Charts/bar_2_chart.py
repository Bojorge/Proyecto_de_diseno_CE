import matplotlib.pyplot as plt
import numpy as np

def plot_message_passing(libraries, ram_usage_1, ram_usage_2, user_cpu_usage_1, user_cpu_usage_2, 
                         system_cpu_usage_1, system_cpu_usage_2, execution_time_1, execution_time_2):
    
    # Número de bibliotecas
    num_libraries = len(libraries)
    bar_width = 0.35  # Ancho de cada barra
    index = np.arange(num_libraries)  # Posiciones de las barras
    
    # Crear figura y ejes
    fig, axs = plt.subplots(4, 1, figsize=(12, 12))  
    
    # Gráfico de uso de RAM
    axs[0].bar(index, ram_usage_1, bar_width, label='SERVER', color='lightblue')
    axs[0].bar(index + bar_width, ram_usage_2, bar_width, label='CLIENT', color='blue')
    axs[0].set_ylabel('RAM (KB)')
    axs[0].set_xticks(index + bar_width / 2)
    axs[0].set_xticklabels(libraries, rotation=45)
    axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')  # Colocar leyenda fuera del gráfico
    
    # Gráfico de uso de CPU en modo usuario
    axs[1].bar(index, user_cpu_usage_1, bar_width, label='SERVER', color='lightgreen')
    axs[1].bar(index + bar_width, user_cpu_usage_2, bar_width, label='CLIENT', color='darkgreen')
    axs[1].set_ylabel('CPU user (s)')
    axs[1].set_xticks(index + bar_width / 2)
    axs[1].set_xticklabels(libraries, rotation=45)
    axs[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')  # Colocar leyenda fuera del gráfico
    
    # Gráfico de uso de CPU en modo sistema
    axs[2].bar(index, system_cpu_usage_1, bar_width, label='SERVER', color='salmon')
    axs[2].bar(index + bar_width, system_cpu_usage_2, bar_width, label='CLIENT', color='firebrick')
    axs[2].set_ylabel('CPU system (s)')
    axs[2].set_xticks(index + bar_width / 2)
    axs[2].set_xticklabels(libraries, rotation=45)
    axs[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')  # Colocar leyenda fuera del gráfico
    
    # Gráfico de tiempo de ejecución
    axs[3].bar(index, execution_time_1, bar_width, label='SERVER', color='plum')
    axs[3].bar(index + bar_width, execution_time_2, bar_width, label='CLIENT', color='mediumpurple')
    axs[3].set_ylabel('Execution time (s)')
    axs[3].set_xticks(index + bar_width / 2)
    axs[3].set_xticklabels(libraries, rotation=45)
    axs[3].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')  # Colocar leyenda fuera del gráfico
    
    # Ajustar márgenes y el espacio entre subgráficos
    plt.subplots_adjust(hspace=0.65, right=0.75)  # Espacio suficiente para la leyenda
    
    # Título general
    fig.suptitle('Sockets', fontsize=15)
    
    # Mostrar gráficos
    plt.show()

# Datos proporcionados para Message Passing
libraries = ['Boost', 'POCO', 'POSIX', 'ZeroMQ']

ram_usage_server = [4596, 6892, 3200, 5888]  # en KB para SERVER
ram_usage_client = [3584, 6016, 3584, 5888]  # en KB para CLIENT

user_cpu_usage_server = [0.025421, 0.027689, 0.011867, 0.212619]  # en segundos para SERVER
user_cpu_usage_client = [0.053765, 0.027568, 0.021672, 0.3989]  # en segundos para CLIENT

system_cpu_usage_server = [0.186097, 0.088606, 0.069026, 0.38085]  # en segundos para SERVER
system_cpu_usage_client = [0.163579, 0.119145, 0.121108, 0.693523]  # en segundos para CLIENT

execution_time_server = [0.589256, 1.37736, 9.52518, 1.86166]  # en segundos para SERVER
execution_time_client = [0.412029, 1.24333, 9.32187, 1.76179]  # en segundos para CLIENT

# Crear gráfica
plot_message_passing(libraries, ram_usage_server, ram_usage_client, user_cpu_usage_server, user_cpu_usage_client, 
                     system_cpu_usage_server, system_cpu_usage_client, execution_time_server, execution_time_client)
