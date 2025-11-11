# 🤖 Reconhecimento Facial para 4 Pessoas

Este repositório reúne o notebook desenvolvido em sala de aula para montar um pipeline completo de **reconhecimento facial** capaz de identificar você e três colegas. O fluxo combina **detecção/alinhamento**, **extração de embeddings** e **classificação supervisionada**, oferecendo ainda uma opção alternativa baseada em *transfer learning* com Keras.

## 🧭 Visão Geral do Pipeline

| Etapa | Descrição |
| --- | --- |
| 📥 Coleta de dados | Organize as fotos em `dados_brutos/`, com uma pasta por pessoa (idealmente 50+ imagens variadas). |
| 🎯 Detecção & alinhamento | A rede **MTCNN** (via `facenet-pytorch`) localiza e alinha as faces, salvando-as em `dados_processados/`. |
| 🧬 Embeddings faciais | O modelo **InceptionResnetV1** gera vetores de características (512 dimensões) a partir das faces alinhadas. |
| 🧠 Classificador SVM | Um **SVM com kernel RBF** aprende a separar os embeddings de cada pessoa. |
| 📊 Avaliação | Relatórios de classificação e matriz de confusão ajudam a validar o desempenho. |
| 📹 Demo em tempo real | A função `webcam_demo()` usa a webcam para reconhecer rostos em tempo real, com limiar para detectar desconhecidos. |

> 💡 **Por que esta abordagem?** Para conjuntos de dados pequenos, extrair embeddings pré-treinados e treinar um classificador leve costuma oferecer mais estabilidade do que treinar uma CNN do zero.

## 🗂️ Estrutura recomendada das pastas

```text
dados_brutos/
├── pessoa1_voce/
│   ├── img001.jpg
│   └── ...
├── pessoa2_colegaA/
├── pessoa3_colegaB/
└── pessoa4_colegaC/
```

Durante o processamento, o notebook cria `dados_processados/` com as faces alinhadas e balanceadas para o treinamento.

## 🛠️ Dependências principais

- Python 3.10+
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch) (MTCNN + InceptionResnetV1)
- PyTorch (com suporte opcional a CUDA)
- OpenCV, NumPy, Pandas, Matplotlib
- scikit-learn (SVM, métricas)
- pillow/pillow-heif (leitura de imagens HEIC/HEIF)
- TensorFlow + Keras *(apenas se optar pela alternativa MobileNetV2)*

> ✅ O notebook inclui células com `pip install` para facilitar a instalação direta em ambientes como Google Colab ou Jupyter local.

## 🚀 Passo a passo rápido

1. **Clone ou baixe** este repositório.
2. **Abra o notebook** `Reconhecimento_Facial_4_Pessoas_2.ipynb` em um ambiente Jupyter (Colab, VS Code, JupyterLab...).
3. **Instale as dependências** executando a seção `Instalação de dependências` apenas se necessário.
4. **Configure os caminhos** para as pastas `dados_brutos/` e `dados_processados/` na seção correspondente.
5. **Execute as células em ordem**, garantindo que a detecção/alinhamento rode antes do treinamento.
6. **Treine e avalie** o classificador, ajustando hiperparâmetros (por exemplo, `C`, `gamma` do SVM) se quiser otimizar o desempenho.
7. **Teste a webcam** chamando `webcam_demo()` para verificar o reconhecimento em tempo real (pressione `q` para sair).

## 🧪 Avaliação e monitoramento

- 📈 Use o `classification_report` para checar precisão, revocação e F1-score de cada pessoa.
- 🧾 A matriz de confusão (`ConfusionMatrixDisplay`) revela quais identidades estão sendo confundidas.
- 🎚️ Ajuste o `threshold` do `webcam_demo()` (valor padrão 0.70) para calibrar a detecção de desconhecidos.

## 🔄 Opção B — Transfer Learning com MobileNetV2

Se quiser experimentar outra abordagem, o notebook traz uma seção comentada com:

- Carregamento das imagens via `tf.data`.
- Backbone **MobileNetV2** pré-treinado no ImageNet.
- Cabeça *Dense* com *softmax* para classificação direta.

Essa opção pode ser útil se o dataset crescer bastante, mas requer mais dados para generalizar bem.

## 🧰 Dicas e boas práticas

- Colete dados com **variação de iluminação, ângulos e expressões**.
- Misture fotos com e sem óculos, mas mantenha diversidade.
- Revise manualmente as faces alinhadas e descarte recortes ruins.
- Considere aumentar o `margin` do MTCNN se partes do rosto estiverem sendo cortadas.
- Faça *data augmentation* leve (flip horizontal, leve ajuste de brilho) caso queira melhorar robustez.

## 🧪 Possíveis extensões

- 📚 Adicionar mais pessoas com re-treinamento incremental do SVM.
- 🔐 Integrar autenticação em aplicações web ou desktop.
- 🛰️ Utilizar câmeras IP/RTSP substituindo a captura da webcam.
- 📦 Exportar o classificador (`joblib.dump`) e carregá-lo em um serviço de inferência.

## 👩‍🏫 Créditos

Projeto desenvolvido durante as aulas de IA/Visão Computacional, adaptado a partir do notebook `"[ Aula 17 ] - CKP04.ipynb"`. Sinta-se à vontade para reutilizar, adaptar e compartilhar com a turma!

---

💬 **Dúvidas?** Abra uma *issue* ou envie o stacktrace do erro para que possamos ajudar.
