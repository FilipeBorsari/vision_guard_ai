#!/usr/bin/env python3
"""
Script de teste rápido para cadastro
"""

import cv2
import time

# Garantir que não há janelas OpenCV abertas
cv2.destroyAllWindows()
time.sleep(0.2)

from main import SistemaReconhecimentoFacial

print("=== TESTE DE CADASTRO ===")
print("Este script vai testar apenas a função de cadastro")
print()

sistema = SistemaReconhecimentoFacial()

nome = input("Digite o nome da pessoa para cadastrar: ").strip()

if nome:
    print(f"\nIniciando cadastro de: {nome}")
    print("Uma janela de vídeo será aberta")
    print("Pressione ESPAÇO quando estiver pronto")
    print("Pressione Q para cancelar")
    print()
    
    # Garantir limpeza antes
    cv2.destroyAllWindows()
    time.sleep(0.2)
    
    sucesso = sistema.adicionar_pessoa(nome)
    
    # Garantir limpeza depois
    cv2.destroyAllWindows()
    time.sleep(0.2)
    
    if sucesso:
        print("\n✅ Cadastro realizado com sucesso!")
        print(f"Total de pessoas cadastradas: {len(sistema.nomes_conhecidos)}")
    else:
        print("\n❌ Cadastro não realizado")
else:
    print("❌ Nome inválido")

# Limpeza final
cv2.destroyAllWindows()
