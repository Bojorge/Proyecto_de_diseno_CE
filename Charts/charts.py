import matplotlib.pyplot as plt

def plot_usage(libraries, ram_usage, user_cpu_usage, system_cpu_usage):
    # Número de bibliotecas
    num_libraries = len(libraries)
    
    # Crear figura y ejes con un tamaño de figura más grande
    fig, axs = plt.subplots(3, 1, figsize=(12, 30))  # Aumenta el tamaño de la figura
    
    # Gráfico de uso de RAM
    axs[0].bar(libraries, ram_usage, color='skyblue')
    axs[0].set_ylabel('Uso de RAM (KB)')
    axs[0].tick_params(axis='x', rotation=45)
    
    # Gráfico de uso de CPU en modo usuario
    axs[1].bar(libraries, user_cpu_usage, color='lightgreen')
    axs[1].set_ylabel('CPU modo usuario (s)')
    axs[1].tick_params(axis='x', rotation=45)
    
    # Gráfico de uso de CPU en modo sistema
    axs[2].bar(libraries, system_cpu_usage, color='salmon')
    axs[2].set_ylabel('CPU modo sistema (s)')
    axs[2].tick_params(axis='x', rotation=45)
    
    # Ajustar el espaciado para que no se sobrepongan las etiquetas
    plt.subplots_adjust(hspace=0.9, top=0.93, bottom=0.12, left=0.1, right=0.95)
    
    # Título de la ventana
    fig.suptitle('Sockets', fontsize=15)
    
    # Mostrar gráficos
    plt.show()

# Ejemplo de uso
libraries = ['POSIX', 'Boost.Asio', 'POCO']
ram_usage = [0, 0, 0]  # en KB
user_cpu_usage = [0.0, 0.0, 0.0]  # en segundos
system_cpu_usage = [0.0, 0.0, 0.0]  # en segundos

plot_usage(libraries, ram_usage, user_cpu_usage, system_cpu_usage)
