# v++ -c -t hw --platform <la_plataforma> -k matrix_mult -o matrix_mult.xo matrix_mult.cpp

# ================================================================
# Makefile para la síntesis y compilación utilizando Vitis HLS y v++.
#
# Este Makefile automatiza los siguientes pasos:
# 1. Síntesis del kernel con Vitis HLS.
# 2. Compilación del kernel con v++ para generar el objeto (.xo).
# 3. Enlazado del objeto compilado para generar el binario ejecutable (.xclbin).
#
# ================================================================

# Para obtener la lista de plataformas disponibles ejecutar: 'v++ --list_platforms'
PLATFORM := xilinx_u200_xdma_201830_2

# Nombre del kernel
KERNEL_NAME := matrix_mult

# Directorios
SRC_DIR := src            # Directorio donde se encuentran los archivos fuente
BUILD_DIR := build        # Directorio donde se generarán los archivos compilados
HLS_SCRIPT := hls_script.tcl  # Script TCL para la síntesis con Vitis HLS

# Archivos Fuente
KERNEL_SRC := $(SRC_DIR)/$(KERNEL_NAME).cpp  # Código fuente del kernel en C++

# Archivos Generados
XO_FILE := $(BUILD_DIR)/$(KERNEL_NAME).xo        # Archivo objeto compilado del kernel
XCLBIN_FILE := $(BUILD_DIR)/$(KERNEL_NAME).xclbin # Archivo binario ejecutable para la FPGA

# Opciones de Compilación
VPP_FLAGS := --platform $(PLATFORM)             # Especifica la plataforma objetivo
VPP_COMP_FLAGS := -c -k $(KERNEL_NAME) -I$(SRC_DIR) # Opciones para la compilación
VPP_LINK_FLAGS := -l                            # Opciones para el enlazado

# Comandos
VITIS_HLS := vitis_hls    # Comando para ejecutar Vitis HLS
VPP := v++                # Comando para ejecutar el compilador v++

# -------------------------------
# Reglas del Makefile
# -------------------------------

# Regla por defecto: Ejecuta todos los pasos
all: $(XCLBIN_FILE)

# Regla para la síntesis HLS
hls:
	@echo "===> Ejecutando síntesis HLS..."
	$(VITIS_HLS) -f $(HLS_SCRIPT)
	@echo "===> Síntesis HLS completada."

# Regla para la compilación del kernel
$(XO_FILE): hls
	@echo "===> Compilando el kernel con v++..."
	$(VPP) $(VPP_FLAGS) $(VPP_COMP_FLAGS) -o $(XO_FILE) $(KERNEL_SRC)
	@echo "===> Compilación del kernel completada. Archivo generado: $(XO_FILE)"

# Regla para el enlazado y generación del binario ejecutable
$(XCLBIN_FILE): $(XO_FILE)
	@echo "===> Enlazando y generando el archivo binario ejecutable..."
	$(VPP) $(VPP_FLAGS) $(VPP_LINK_FLAGS) -o $(XCLBIN_FILE) $(XO_FILE)
	@echo "===> Enlazado completado. Archivo binario generado: $(XCLBIN_FILE)"

# Regla para limpiar los archivos generados
clean:
	@echo "===> Limpiando archivos generados..."
	rm -rf $(BUILD_DIR) *.log *.jou *.dir
	@echo "===> Limpieza completada."

# Regla de ayuda
help:
	@echo "Uso del Makefile:"
	@echo "  make all       - Ejecuta todo el flujo de síntesis y compilación."
	@echo "  make hls       - Ejecuta solo la síntesis HLS."
	@echo "  make clean     - Elimina todos los archivos generados."
	@echo "  make help      - Muestra esta ayuda."

# Crea el directorio de compilación si no existe
$(shell mkdir -p $(BUILD_DIR))
