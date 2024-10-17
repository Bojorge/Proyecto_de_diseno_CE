import matplotlib.pyplot as plt

def plot_performance_metrics(execution_numbers, ram_usage, user_cpu_usage, system_cpu_usage, latency):
    # Crear figura y ejes
    fig, axs = plt.subplots(4, 1, figsize=(10, 20))
    
    # Gráfico de uso de RAM
    axs[0].plot(execution_numbers, ram_usage, marker='o', color='skyblue')
    axs[0].set_xlabel('Execution #')
    axs[0].set_ylabel('RAM (KB)')
    axs[0].set_title('RAM Usage per Execution')
    axs[0].grid(True)

    # Gráfico de uso de CPU en modo usuario
    axs[1].plot(execution_numbers, user_cpu_usage, marker='o', color='lightgreen')
    axs[1].set_xlabel('Execution #')
    axs[1].set_ylabel('User Mode (s)')
    axs[1].set_title('User Mode CPU Usage per Execution')
    axs[1].grid(True)

    # Gráfico de uso de CPU en modo sistema
    axs[2].plot(execution_numbers, system_cpu_usage, marker='o', color='salmon')
    axs[2].set_xlabel('Execution #')
    axs[2].set_ylabel('System Mode (s)')
    axs[2].set_title('System Mode CPU Usage per Execution')
    axs[2].grid(True)

    # Gráfico de latencia
    axs[3].plot(execution_numbers, latency, marker='o', color='mediumpurple')
    axs[3].set_xlabel('Execution #')
    axs[3].set_ylabel('Latency (s)')
    axs[3].set_title('Latency per Execution')
    axs[3].grid(True)

    # Ajustar el espaciado entre los gráficos
    plt.tight_layout()
    
    # Ajustes
    plt.subplots_adjust(hspace=1.5, top=0.85, bottom=0.12, left=0.1, right=0.95)
    
    # Título principal
    fig.suptitle('Sockets (server) - Boost/Asio', fontsize=16)
    
    # Mostrar gráficos
    plt.show()

# Datos proporcionados
execution_numbers = list(range(1, 11))  # Lista de números de ejecución del 1 al 10
ram_usage = [3836, 4840, 3964, 4464, 4092, 4216, 6652, 6136, 4480, 4348]
system_cpu_usage = [0.147054, 0.177596, 0.15278, 0.157008, 0.15925, 0.157813, 0.16888, 0.166934, 0.153216, 0.15607]
user_cpu_usage = [0.026769, 0.021359, 0.030278, 0.044902, 0.02222, 0.026302, 0.026454, 0.039608, 0.016269, 0.029885]
latency = [0.681458, 0.478833, 0.595089, 0.719109, 0.567252, 0.560148, 0.626429, 0.49481, 0.449764, 0.537526]

plot_performance_metrics(execution_numbers, ram_usage, user_cpu_usage, system_cpu_usage, latency)
