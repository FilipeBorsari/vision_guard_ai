# Sistema de Reconhecimento Facial para Condomínio

Sistema simples de reconhecimento facial em Python usando `face_recognition` e OpenCV para controle de acesso em condomínios.

## 🎯 Funcionalidades

- **Reconhecimento facial em tempo real** via webcam
- **Cadastro de moradores** com captura de foto pela webcam
- **Registro de acessos** com data, hora e nível de confiança
- **Sistema de cache** para carregar encodings rapidamente
- **Interface interativa** via menu no terminal
- **Logs em JSON** organizados por data
- **Screenshots** durante o reconhecimento

## 📋 Requisitos

- Python 3.7+
- Webcam
- Linux/Windows/MacOS

## 🚀 Instalação

### 1. Clone ou baixe o projeto

```bash
cd /home/filipe-borsari/Desktop/projeto_condominio
```

### 2. Crie um ambiente virtual (recomendado)

```bash
python3 -m venv .venv
source .venv/bin/activate  # No Linux/Mac
# .venv\Scripts\activate   # No Windows
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

**Nota:** A instalação do `dlib` pode demorar alguns minutos e requer cmake e compiladores C++.

#### Problemas com dlib no Linux?
```bash
sudo apt-get install cmake build-essential
pip install dlib
```

#### Problemas com dlib no Windows?
Baixe o wheel pré-compilado em: https://github.com/z-mahmud22/Dlib_Windows_Python3.x

## 📖 Como Usar

### Modo 1: Menu Interativo

```bash
python main.py
```

O menu oferece as seguintes opções:
1. **Carregar pessoas cadastradas** - Processa imagens da pasta `imagens_conhecidas/`
2. **Iniciar reconhecimento facial** - Ativa a webcam para reconhecimento em tempo real
3. **Cadastrar nova pessoa** - Captura foto pela webcam e adiciona ao sistema
4. **Ver estatísticas** - Mostra pessoas cadastradas e acessos do dia
5. **Sair** - Encerra o sistema

### Modo 2: Uso Programático

```python
from main import SistemaReconhecimentoFacial

# Criar instância
sistema = SistemaReconhecimentoFacial()

# Carregar imagens conhecidas
sistema.carregar_imagens_conhecidas()

# Iniciar reconhecimento
sistema.iniciar_reconhecimento()
```

## 📁 Estrutura do Projeto

```
projeto_condominio/
│
├── main.py                    # Código principal
├── requirements.txt           # Dependências
├── README.md                  # Este arquivo
│
├── imagens_conhecidas/        # Imagens dos moradores
│   ├── joao_silva.jpg
│   ├── maria_santos.jpg
│   └── ...
│
├── logs/                      # Logs de acesso e screenshots
│   ├── acessos_2025-11-23.json
│   ├── screenshot_20251123_143022.jpg
│   └── ...
│
└── encodings.pkl             # Cache dos encodings faciais
```

## 👥 Cadastrando Moradores

### Método 1: Via Menu (Recomendado)
1. Execute `python main.py`
2. Escolha opção 3 (Cadastrar nova pessoa)
3. Digite o nome
4. Posicione-se em frente à webcam
5. Pressione ESPAÇO para capturar

### Método 2: Adicionar Imagens Manualmente
1. Tire uma foto clara do rosto da pessoa
2. Salve na pasta `imagens_conhecidas/` com o nome da pessoa
   - Exemplo: `joao_silva.jpg`, `maria_santos.jpg`
3. Execute opção 1 do menu para processar

**Dicas para melhores resultados:**
- Foto com boa iluminação
- Rosto centralizado e visível
- Evite óculos escuros ou acessórios que cubram o rosto
- Uma foto por pessoa

## 🎮 Controles Durante Reconhecimento

- **q** - Sair do reconhecimento
- **s** - Tirar screenshot (salvo em `logs/`)

## 📊 Sistema de Logs

Os acessos são registrados em arquivos JSON na pasta `logs/`:

```json
[
  {
    "nome": "joao_silva",
    "data_hora": "2025-11-23 14:30:22",
    "confianca": "95.80%"
  }
]
```

## ⚙️ Configuração Avançada

### Ajustar tolerância do reconhecimento

No arquivo `main.py`, linha ~165:

```python
matches = face_recognition.compare_faces(
    self.encodings_conhecidos, 
    face_encoding, 
    tolerance=0.6  # Menor = mais rigoroso (0.4-0.6 recomendado)
)
```

### Mostrar/ocultar porcentagem de confiança

```python
sistema.iniciar_reconhecimento(mostrar_confianca=True)  # Padrão
sistema.iniciar_reconhecimento(mostrar_confianca=False) # Ocultar
```

## 🔧 Solução de Problemas

### Webcam não abre
- Verifique se outra aplicação está usando a câmera
- Tente trocar `cv2.VideoCapture(0)` para `cv2.VideoCapture(1)`

### "Nenhum rosto encontrado"
- Melhore a iluminação
- Certifique-se de que o rosto está visível e frontal
- Reduza a distância da câmera

### Performance lenta
- O sistema já processa frames alternados para melhor performance
- Reduza a resolução da webcam se necessário

### Erro ao instalar dlib
- Certifique-se de ter cmake instalado: `pip install cmake`
- No Linux: `sudo apt-get install cmake build-essential`
- Use wheels pré-compilados quando disponíveis

## 📝 Licença

Este projeto é de código aberto para fins educacionais.

## 🤝 Contribuições

Sugestões e melhorias são bem-vindas!

## 📧 Suporte

Para problemas ou dúvidas, abra uma issue no repositório do projeto.
