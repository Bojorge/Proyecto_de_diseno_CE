#!/bin/bash
./writer &  # Ejecuta el escritor en segundo plano
./reader    # Ejecuta el lector en primer plano
wait        # Espera a que terminen ambos procesos
