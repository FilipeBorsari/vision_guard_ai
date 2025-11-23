"""
Sistema de Reconhecimento Facial para Condomínio
Utiliza face_recognition e OpenCV para identificar moradores
"""

import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime
import pickle
import json


class SistemaReconhecimentoFacial:
    def __init__(self, pasta_imagens="imagens_conhecidas", pasta_logs="logs"):
        """
        Inicializa o sistema de reconhecimento facial
        
        Args:
            pasta_imagens: Pasta contendo imagens dos moradores conhecidos
            pasta_logs: Pasta para salvar logs de acesso
        """
        self.pasta_imagens = pasta_imagens
        self.pasta_logs = pasta_logs
        self.encodings_conhecidos = []
        self.nomes_conhecidos = []
        self.arquivo_encodings = "encodings.pkl"
        
        # Criar pastas se não existirem
        os.makedirs(pasta_imagens, exist_ok=True)
        os.makedirs(pasta_logs, exist_ok=True)
        
    def carregar_imagens_conhecidas(self):
        """
        Carrega e processa imagens da pasta de imagens conhecidas
        O nome do arquivo (sem extensão) será usado como nome da pessoa
        """
        print("Carregando imagens conhecidas...")
        
        if os.path.exists(self.arquivo_encodings):
            print("Carregando encodings salvos...")
            with open(self.arquivo_encodings, 'rb') as f:
                data = pickle.load(f)
                self.encodings_conhecidos = data['encodings']
                self.nomes_conhecidos = data['nomes']
                print(f"✓ {len(self.nomes_conhecidos)} pessoas carregadas do cache")
                return
        
        # Se não houver cache, processar imagens
        for arquivo in os.listdir(self.pasta_imagens):
            if arquivo.lower().endswith(('.jpg', '.jpeg', '.png')):
                caminho = os.path.join(self.pasta_imagens, arquivo)
                nome = os.path.splitext(arquivo)[0]
                
                try:
                    # Carregar imagem
                    imagem = face_recognition.load_image_file(caminho)
                    
                    # Obter encoding do rosto
                    encodings = face_recognition.face_encodings(imagem)
                    
                    if encodings:
                        self.encodings_conhecidos.append(encodings[0])
                        self.nomes_conhecidos.append(nome)
                        print(f"✓ {nome} carregado com sucesso")
                    else:
                        print(f"✗ Nenhum rosto encontrado em {arquivo}")
                        
                except Exception as e:
                    print(f"✗ Erro ao processar {arquivo}: {e}")
        
        # Salvar encodings para uso futuro
        if self.encodings_conhecidos:
            with open(self.arquivo_encodings, 'wb') as f:
                pickle.dump({
                    'encodings': self.encodings_conhecidos,
                    'nomes': self.nomes_conhecidos
                }, f)
            print(f"\n✓ Total de {len(self.nomes_conhecidos)} pessoas cadastradas")
        else:
            print("\n⚠ Nenhuma pessoa foi cadastrada!")
            print(f"Adicione imagens na pasta '{self.pasta_imagens}'")
    
    def registrar_acesso(self, nome, confianca):
        """
        Registra o acesso de uma pessoa no arquivo de log
        
        Args:
            nome: Nome da pessoa identificada
            confianca: Nível de confiança da identificação
        """
        timestamp = datetime.now()
        data_arquivo = timestamp.strftime("%Y-%m-%d")
        arquivo_log = os.path.join(self.pasta_logs, f"acessos_{data_arquivo}.json")
        
        registro = {
            "nome": nome,
            "data_hora": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "confianca": f"{confianca:.2%}"
        }
        
        # Carregar logs existentes ou criar novo
        logs = []
        if os.path.exists(arquivo_log):
            with open(arquivo_log, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        
        logs.append(registro)
        
        # Salvar logs atualizados
        with open(arquivo_log, 'w', encoding='utf-8') as f:
            json.dump(logs, indent=2, ensure_ascii=False, fp=f)
    
    def iniciar_reconhecimento(self, mostrar_confianca=True):
        """
        Inicia o reconhecimento facial em tempo real usando a webcam
        
        Args:
            mostrar_confianca: Se True, mostra o percentual de confiança na tela
        """
        if not self.encodings_conhecidos:
            print("⚠ Nenhuma pessoa cadastrada! Execute carregar_imagens_conhecidas() primeiro.")
            return
        
        # Limpar qualquer janela anterior
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        
        video_capture = None
        
        try:
            # Inicializar webcam
            video_capture = cv2.VideoCapture(0)
            
            if not video_capture.isOpened():
                print("✗ Erro ao acessar a webcam!")
                return
            
            print("\n=== Sistema de Reconhecimento Facial Ativo ===")
            print("Pressione Q para sair")
            print("Pressione S para tirar screenshot")
            print("=" * 46)
            
            # Criar janela uma vez
            window_name = 'Reconhecimento Facial - Condominio'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            
            # Variáveis para controle
            process_this_frame = True
            ultimo_registro = {}
            face_locations = []
            face_names = []
            face_confidences = []
            frame_count = 0
            
            while True:
                # Capturar frame
                ret, frame = video_capture.read()
                
                if not ret:
                    print("✗ Erro ao capturar frame da webcam")
                    break
                
                frame_count += 1
                
                # Processar apenas frames alternados para melhor performance
                if process_this_frame:
                    # Redimensionar frame para processamento mais rápido
                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    
                    # Encontrar rostos e encodings no frame atual
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                    
                    face_names = []
                    face_confidences = []
                    
                    for face_encoding in face_encodings:
                        # Comparar com rostos conhecidos
                        matches = face_recognition.compare_faces(
                            self.encodings_conhecidos, 
                            face_encoding, 
                            tolerance=0.6
                        )
                        name = "Desconhecido"
                        confidence = 0
                        
                        # Calcular distância facial (menor = mais similar)
                        face_distances = face_recognition.face_distance(
                            self.encodings_conhecidos, 
                            face_encoding
                        )
                        
                        if len(face_distances) > 0:
                            best_match_index = np.argmin(face_distances)
                            
                            if matches[best_match_index]:
                                name = self.nomes_conhecidos[best_match_index]
                                confidence = 1 - face_distances[best_match_index]
                                
                                # Registrar acesso (evitar múltiplos registros)
                                agora = datetime.now()
                                if name not in ultimo_registro or \
                                   (agora - ultimo_registro[name]).seconds > 30:
                                    self.registrar_acesso(name, confidence)
                                    ultimo_registro[name] = agora
                                    print(f"✓ Acesso registrado: {name} ({confidence:.1%})")
                        
                        face_names.append(name)
                        face_confidences.append(confidence)
                
                process_this_frame = not process_this_frame
                
                # Desenhar resultados no frame (apenas se houver rostos detectados)
                if face_locations and len(face_locations) == len(face_names) == len(face_confidences):
                    for (top, right, bottom, left), name, confidence in zip(
                        face_locations, face_names, face_confidences
                    ):
                        # Escalar de volta para tamanho original
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4
                        
                        # Definir cor (verde para conhecido, vermelho para desconhecido)
                        color = (0, 255, 0) if name != "Desconhecido" else (0, 0, 255)
                        
                        # Desenhar retângulo ao redor do rosto
                        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                        
                        # Desenhar caixa com nome
                        cv2.rectangle(
                            frame, 
                            (left, bottom - 35), 
                            (right, bottom), 
                            color, 
                            cv2.FILLED
                        )
                        
                        # Preparar texto
                        if mostrar_confianca and confidence > 0:
                            texto = f"{name} ({confidence:.0%})"
                        else:
                            texto = name
                        
                        cv2.putText(
                            frame, 
                            texto, 
                            (left + 6, bottom - 6), 
                            cv2.FONT_HERSHEY_DUPLEX, 
                            0.6, 
                            (255, 255, 255), 
                            1
                        )
                
                # Mostrar frame UMA VEZ por iteração
                cv2.imshow(window_name, frame)
                
                # Controles do teclado (aumentado para 30ms)
                key = cv2.waitKey(30) & 0xFF
                
                if key == ord('q') or key == ord('Q') or key == 27:  # Q ou ESC
                    print("\n✓ Sistema encerrado pelo usuário")
                    break
                elif key == ord('s') or key == ord('S'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    screenshot_path = os.path.join(self.pasta_logs, f"screenshot_{timestamp}.jpg")
                    cv2.imwrite(screenshot_path, frame)
                    print(f"✓ Screenshot salvo: {screenshot_path}")
        
        except Exception as e:
            print(f"✗ Erro durante reconhecimento: {e}")
        
        finally:
            # SEMPRE liberar recursos
            if video_capture is not None:
                video_capture.release()
            
            cv2.destroyAllWindows()
            
            # Aguardar fechamento completo
            for _ in range(5):
                cv2.waitKey(1)
    
    def adicionar_pessoa(self, nome, usar_webcam=True, caminho_imagem=None):
        """
        Adiciona uma nova pessoa ao sistema
        
        Args:
            nome: Nome da pessoa
            usar_webcam: Se True, captura foto da webcam. Se False, usa caminho_imagem
            caminho_imagem: Caminho para imagem existente (usado se usar_webcam=False)
        """
        if usar_webcam:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            
            video_capture = None
            cadastro_sucesso = False
            
            try:
                video_capture = cv2.VideoCapture(0)
                
                if not video_capture.isOpened():
                    print("✗ Erro ao acessar a webcam!")
                    return False
                
                print(f"\n=== Cadastrando: {nome} ===")
                print("Pressione ESPAÇO para capturar a foto")
                print("Pressione Q para cancelar")
                print()
                
                window_name = 'Cadastro de Pessoa'
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                
                frame_count = 0
                face_locations = []  # Inicializar variável
                
                while True:
                    ret, frame = video_capture.read()
                    if not ret:
                        print("✗ Erro ao capturar frame da webcam")
                        break
                    
                    frame_count += 1
                    
                    # Processar detecção a cada 3 frames para melhor performance
                    if frame_count % 3 == 0:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        face_locations = face_recognition.face_locations(rgb_frame)
                    
                    # Desenhar retângulos e informações
                    if face_locations:
                        for (top, right, bottom, left) in face_locations:
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                            cv2.putText(frame, "Rosto OK!", (left, top - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Nenhum rosto detectado", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # Instruções
                    cv2.putText(frame, f"Cadastrando: {nome}", (10, frame.shape[0] - 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, "ESPACO = Capturar | Q = Sair", (10, frame.shape[0] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Mostrar frame UMA VEZ por iteração
                    cv2.imshow(window_name, frame)
                    
                    # Esperar pela tecla (aumentado para 50ms)
                    key = cv2.waitKey(50) & 0xFF
                    
                    if key == ord(' ') or key == 32:  # ESPAÇO
                        if face_locations:
                            caminho_salvar = os.path.join(self.pasta_imagens, f"{nome}.jpg")
                            cv2.imwrite(caminho_salvar, frame)
                            print(f"✓ Foto salva: {caminho_salvar}")
                            
                            # Processar encoding
                            encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                            if encodings:
                                self.encodings_conhecidos.append(encodings[0])
                                self.nomes_conhecidos.append(nome)
                                
                                # Atualizar cache
                                with open(self.arquivo_encodings, 'wb') as f:
                                    pickle.dump({
                                        'encodings': self.encodings_conhecidos,
                                        'nomes': self.nomes_conhecidos
                                    }, f)
                                
                                print(f"✓ {nome} cadastrado com sucesso!")
                                cadastro_sucesso = True
                                break
                        else:
                            print("✗ Nenhum rosto detectado! Posicione-se melhor.")
                    
                    elif key == ord('q') or key == ord('Q') or key == 27:  # Q ou ESC
                        print("✗ Cadastro cancelado pelo usuário")
                        break
            
            except Exception as e:
                print(f"✗ Erro durante cadastro: {e}")
            
            finally:
                # SEMPRE liberar recursos
                if video_capture is not None:
                    video_capture.release()
                
                cv2.destroyAllWindows()
                
                # Aguardar fechamento completo
                for _ in range(5):
                    cv2.waitKey(1)
            
            return cadastro_sucesso
        
        return False


def menu_principal():
    """Menu interativo do sistema"""
    sistema = SistemaReconhecimentoFacial()
    
    while True:
        print("\n" + "=" * 50)
        print("  SISTEMA DE RECONHECIMENTO FACIAL - CONDOMÍNIO")
        print("=" * 50)
        print("1. Carregar pessoas cadastradas")
        print("2. Iniciar reconhecimento facial")
        print("3. Cadastrar nova pessoa")
        print("4. Ver estatísticas")
        print("5. Sair")
        print("=" * 50)
        
        opcao = input("Escolha uma opção: ").strip()
        
        if opcao == "1":
            sistema.carregar_imagens_conhecidas()
        
        elif opcao == "2":
            if not sistema.encodings_conhecidos:
                print("\n⚠ Carregue as pessoas cadastradas primeiro (opção 1)")
            else:
                # Garantir que não há janelas abertas
                cv2.destroyAllWindows()
                cv2.waitKey(100)
                
                sistema.iniciar_reconhecimento()
                
                # Garantir limpeza após reconhecimento
                cv2.destroyAllWindows()
                cv2.waitKey(100)
        
        elif opcao == "3":
            # Garantir que não há janelas abertas antes de começar cadastro
            cv2.destroyAllWindows()
            cv2.waitKey(100)
            
            nome = input("Nome da pessoa: ").strip()
            if nome:
                sucesso = sistema.adicionar_pessoa(nome)
                # Garantir limpeza após cadastro
                cv2.destroyAllWindows()
                cv2.waitKey(100)
                
                if not sucesso:
                    print("⚠ Cadastro não foi concluído")
            else:
                print("✗ Nome inválido")
        
        elif opcao == "4":
            print(f"\n=== Estatísticas ===")
            print(f"Pessoas cadastradas: {len(sistema.nomes_conhecidos)}")
            if sistema.nomes_conhecidos:
                print("Lista:", ", ".join(sistema.nomes_conhecidos))
            
            # Contar acessos de hoje
            hoje = datetime.now().strftime("%Y-%m-%d")
            arquivo_log = os.path.join(sistema.pasta_logs, f"acessos_{hoje}.json")
            if os.path.exists(arquivo_log):
                with open(arquivo_log, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
                    print(f"Acessos hoje: {len(logs)}")
            else:
                print("Acessos hoje: 0")
        
        elif opcao == "5":
            print("\n✓ Encerrando sistema...")
            break
        
        else:
            print("\n✗ Opção inválida!")


if __name__ == "__main__":
    # Você pode executar o menu interativo ou usar o sistema programaticamente
    
    # Opção 1: Menu interativo
    menu_principal()
    
    # Opção 2: Uso programático (descomente para usar)
    # sistema = SistemaReconhecimentoFacial()
    # sistema.carregar_imagens_conhecidas()
    # sistema.iniciar_reconhecimento()
