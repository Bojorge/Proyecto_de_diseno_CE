import matplotlib.pyplot as plt

def plot_performance_metrics(execution_numbers, ram_usage, user_cpu_usage, system_cpu_usage, latency):
    # Crear figura y ejes
    fig, axs = plt.subplots(4, 1, figsize=(10, 20))
    
    # Gráfico de uso de RAM
    axs[0].plot(execution_numbers, ram_usage, marker='o', color='skyblue')
    axs[0].set_xlabel('Número de Ejecución')
    axs[0].set_ylabel('RAM (KB)')
    axs[0].set_title('Uso de RAM por Ejecución')
    ##axs[0].legend()
    axs[0].grid(True)

    # Gráfico de uso de CPU en modo usuario
    axs[1].plot(execution_numbers, user_cpu_usage, marker='o', color='lightgreen')
    axs[1].set_xlabel('Número de Ejecución')
    axs[1].set_ylabel('CPU Usuario (s)')
    axs[1].set_title('Uso de CPU en Modo Usuario por Ejecución')
    ##axs[1].legend()
    axs[1].grid(True)

    # Gráfico de uso de CPU en modo sistema
    axs[2].plot(execution_numbers, system_cpu_usage, marker='o', color='salmon')
    axs[2].set_xlabel('Número de Ejecución')
    axs[2].set_ylabel('CPU Sistema (s)')
    axs[2].set_title('Uso de CPU en Modo Sistema por Ejecución')
    ##axs[2].legend()
    axs[2].grid(True)

    # Gráfico de latencia
    axs[3].plot(execution_numbers, latency, marker='o', color='mediumpurple')
    axs[3].set_xlabel('Número de Ejecución')
    axs[3].set_ylabel('Latencia (ms)')
    axs[3].set_title('Latencia por Ejecución')
    ##axs[3].legend()
    axs[3].grid(True)

    # Ajustar el espaciado entre los gráficos
    plt.tight_layout()
    
    # Ajustes
    plt.subplots_adjust(hspace=1.5, top=0.85, bottom=0.12, left=0.1, right=0.95)
    
    # Título principal
    fig.suptitle('Resultados de 10 Ejecuciones', fontsize=16)
    
    # Mostrar gráficos
    plt.show()

# Ejemplo de uso con datos ficticios para 10 ejecuciones
execution_numbers = list(range(1, 11))  # Lista de números de ejecución del 1 al 10
ram_usage = [512, 490, 530, 505, 495, 520, 515, 510, 500, 505]  # Uso de RAM en KB
user_cpu_usage = [1.2, 1.3, 1.1, 1.2, 1.2, 1.3, 1.4, 1.2, 1.1, 1.2]  # CPU en modo usuario en segundos
system_cpu_usage = [0.8, 0.7, 0.9, 0.8, 0.7, 0.8, 0.9, 0.8, 0.7, 0.8]  # CPU en modo sistema en segundos
latency = [100, 105, 98, 102, 100, 99, 103, 101, 100, 104]  # Latencia en milisegundos

plot_performance_metrics(execution_numbers, ram_usage, user_cpu_usage, system_cpu_usage, latency)
