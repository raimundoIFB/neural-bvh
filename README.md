# 🧠 Neural BVH — Bounding Volume Hierarchies com Rede Neural

Este projeto implementa e compara dois métodos para construção de **BVHs (Bounding Volume Hierarchies)** aplicados em computação gráfica, renderização e simulações físicas:

- **BVH tradicional:** baseado na heurística de área superficial (SAH).  
- **BVH neural:** usa uma rede neural supervisionada para prever pontos de divisão (splits) mais eficientes, reduzindo sobreposição (EPO) e custo médio (SAH).

---

## ⚙️ Estrutura do Projeto

```
neural_bvh/
│
├── core/                 # Módulos clássicos (geometria, BVH e métricas)
│   ├── geometry.py
│   ├── bvh.py
│   ├── metrics.py
│   └── utils.py
│
├── neural/               # Módulos da BVH neural
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── dataset.py
│
├── data/                 # Dados e resultados
│   ├── scenes/           # Cenas OBJ geradas
│   ├── models/           # Modelos neurais (.pth)
│   └── results/          # Comparações e gráficos
│
├── main_demo.py          # Pipeline completo: geração, treino e comparação
├── requirements.txt      # Dependências
└── Neural_BVH_Documentacao_Completa.pdf  # Documento técnico detalhado
```

---

## 🚀 Execução Rápida

### 1️⃣ Ative seu ambiente virtual:
```bash
source AmbFernando/bin/activate
```

### 2️⃣ Execute o pipeline completo:
```bash
python main_demo.py
```

### 3️⃣ Saída esperada:
- `data/scenes/scene_toy.obj`
- `data/results/compare_summary.txt`
- `data/results/compare_plot.png`
- `data/results/compare_plot.svg`

---

## 📊 Resultados

O sistema gera gráficos comparando **EPO (Extent of Primitive Overlap)** e **SAH (Surface Area Heuristic)**:

- **Barras azuis:** EPO (quanto menor, melhor)
- **Linha rosa:** SAH médio (quanto menor, melhor)
- Valores reais acima dos marcadores
- Gráficos salvos em `data/results/`

Exemplo de saída:

📈 `data/results/compare_plot.png`

---

## 🧠 Extensões Possíveis

- Suporte a **dados CAD ou objetos reais**
- Treinamento com **arquiteturas mais profundas (MLP/Transformer)**
- Integração com **Blender** ou **Unity**
- Implementação de **BVH paralela (GPU)**
- Comparação de múltiplas cenas automáticas

---

## 📄 Documentação Completa

A documentação técnica detalhada está disponível em:

📘 [Neural_BVH_Documentacao_Completa.pdf](Neural_BVH_Documentacao_Completa.pdf)

---

## 👨‍💻 Autor

**Raimundo C. da Silva Vasconcelos**  
📅 2025  
🔗 Projeto acadêmico para pesquisa e ensino sobre inteligência artificial aplicada à computação gráfica.