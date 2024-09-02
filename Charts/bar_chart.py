import matplotlib.pyplot as plt

def plot_usage(libraries, ram_usage, user_cpu_usage, system_cpu_usage, total_execution_time):
    # Número de bibliotecas
    num_libraries = len(libraries)
    
    # Crear figura y ejes con un tamaño de figura más grande
    fig, axs = plt.subplots(4, 1, figsize=(12, 40))  # Aumenta el tamaño de la figura
    
    # Gráfico de uso de RAM
    axs[0].bar(libraries, ram_usage, color='skyblue')
    axs[0].set_ylabel('Uso de \nRAM (KB)')
    axs[0].tick_params(axis='x', rotation=45)
    
    # Gráfico de uso de CPU en modo usuario
    axs[1].bar(libraries, user_cpu_usage, color='lightgreen')
    axs[1].set_ylabel('CPU modo \nusuario (s)')
    axs[1].tick_params(axis='x', rotation=45)
    
    # Gráfico de uso de CPU en modo sistema
    axs[2].bar(libraries, system_cpu_usage, color='salmon')
    axs[2].set_ylabel('CPU modo \nsistema (s)')
    axs[2].tick_params(axis='x', rotation=45)

    # Gráfico de tiempo total de ejecución
    axs[3].bar(libraries, total_execution_time, color='mediumpurple')
    axs[3].set_ylabel('Tiempo de \nejecución (s)')
    axs[3].tick_params(axis='x', rotation=45)
    
    # Ajustes
    plt.subplots_adjust(hspace=1.5, top=0.93, bottom=0.12, left=0.1, right=0.95)
    
    # Título de la ventana
    fig.suptitle('SHARED MEMORY', fontsize=15)
    
    # Mostrar gráficos
    plt.show()


libraries = ['Boost', 'POCO', 'POSIX']
ram_usage = [3392, 4743, 3264]  # en KB
user_cpu_usage = [0.00607, 0.00558, 0.00381]  # en segundos
system_cpu_usage = [0.00343, 0.00503, 0.00610]  # en segundos
total_execution_time = [10.0176, 10.01465, 10.0161]  # en segundos

plot_usage(libraries, ram_usage, user_cpu_usage, system_cpu_usage, total_execution_time)
