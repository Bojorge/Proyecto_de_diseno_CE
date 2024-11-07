# Setting relevant variables
ROOT_DIR := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))
TARGET := hw
PLATFORM := $(ROOT_DIR)/../../../kv260_custom.xpfm

# Setting input files
HLS_FILE := matrix_mult_BLAS.cpp
HLS_FILE_NAME := matrix_mult_BLAS

# Setting directories
TEMP_DIR := ./tmp.$(TARGET)
BUILD_DIR := ./build.$(TARGET)
PACKAGE_OUT := ./package.$(TARGET)
LINK_OUTPUT := $(BUILD_DIR)/$(HLS_FILE_NAME).link.xclbin
EMCONFIG_DIR := $(TEMP_DIR)

# Packaging
FPGA_BIN := $(PACKAGE_OUT)/$(HLS_FILE_NAME).bit.bin
XCL_BIN := $(PACKAGE_OUT)/$(HLS_FILE_NAME).xclbin
SHELL_JSON := $(PACKAGE_OUT)/shell.json
DTB_FILE := $(PACKAGE_OUT)/$(HLS_FILE_NAME).dtbo

# Setting V++ Flags
VPP_FLAGS += -I/tools/Xilinx-2023.2/Vitis_Libraries/blas/L1/include/hw  # Ruta de BLAS L1 hardware
VPP_FLAGS += -I/tools/Xilinx-2023.2/Vitis_Libraries/blas/L2/include/hw  # Ruta de BLAS L2 hardware 
VPP_FLAGS += -I/tools/Xilinx-2023.2/Vitis_Libraries/blas/L3/include/sw  # Ruta de BLAS L3 software

VPP_FLAGS += --save-temps --jobs 6
VPP_CFLAGS :=
VPP_PFLAGS :=
VPP_LDFLAGS :=

.PHONY: all clean cleanall fpgabin

all: build

.PHONY: build
build: check_platform emconfig $(LINK_OUTPUT) $(FPGA_BIN) $(XCL_BIN)

# Rules for creating the HW
check_platform:
ifndef PLATFORM
	$(error PLATFORM not set. Please set the PLATFORM properly and rerun. Run "make help" for more details.)
endif

.PHONY: hls_build
hls_build: $(TEMP_DIR)/$(HLS_FILE_NAME).xo

$(TEMP_DIR)/$(HLS_FILE_NAME).xo: $(HLS_FILE)
	mkdir -p $(TEMP_DIR)
	v++ -c $(VPP_FLAGS) -t $(TARGET) --platform $(PLATFORM) -k $(HLS_FILE_NAME) --temp_dir $(TEMP_DIR) -I'$(<D)' $(VPP_CFLAGS) -o '$@' '$<'

$(LINK_OUTPUT): $(TEMP_DIR)/$(HLS_FILE_NAME).xo
	mkdir -p $(BUILD_DIR)
	v++ -l $(VPP_FLAGS) $(VPP_LDFLAGS) -t $(TARGET) --platform $(PLATFORM) --temp_dir $(TEMP_DIR) -o '$(LINK_OUTPUT)' $^

emconfig: $(EMCONFIG_DIR)/emconfig.json
$(EMCONFIG_DIR)/emconfig.json:
	emconfigutil --platform $(PLATFORM) --od $(EMCONFIG_DIR)

system.bit: $(LINK_OUTPUT)
	cp $(TEMP_DIR)/link/int/$@ .

bootgen.bif:
	echo "all:{system.bit}" > bootgen.bif

$(FPGA_BIN): system.bit bootgen.bif $(SHELL_JSON) $(DTB_FILE)
	mkdir -p $(PACKAGE_OUT)
	bootgen -w -arch zynqmp -process_bitstream bin -image bootgen.bif
	cp system.bit.bin $@

$(XCL_BIN): $(LINK_OUTPUT)
	mkdir -p $(PACKAGE_OUT)
	cp $< $@

$(SHELL_JSON): $(ROOT_DIR)/../../../shell.json
	mkdir -p $(PACKAGE_OUT)
	cp $< $@

$(DTB_FILE): $(ROOT_DIR)/../../../pl.dtbo
	mkdir -p $(PACKAGE_OUT)
	cp $< $@

# Cleaning rules
clean:
	$(RM) -rf $(EXECUTABLE) *.xclbin/{*sw_emu*,*hw_emu*}
	$(RM) -rf profile_* TempConfig system_estimate.xtxt *.rpt *.csv
	$(RM) -rf src/*.ll *v++* .Xil emconfig.json dltmp* xmltmp* *.log *.jou *.wcfg *.wdb

cleanall: clean
	$(RM) -rf build_dir* sd_card*
	$(RM) -rf package.* tmp.*
	$(RM) -rf _x* *xclbin.run_summary qemu-memory-_* emulation _vimage pl* start_simulation.sh *.xclbin
