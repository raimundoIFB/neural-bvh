neural_bvh/
│
├── core/ # Módulos clássicos (geometria, BVH e métricas)
│ ├── geometry.py
│ ├── bvh.py
│ ├── metrics.py
│ └── utils.py
│
├── neural/ # Módulos da BVH neural
│ ├── model.py
│ ├── train.py
│ ├── evaluate.py
│ └── dataset.py
│
├── data/ # Dados e resultados
│ ├── scenes/ # Cenas OBJ geradas
│ ├── models/ # Modelos neurais (.pth)
│ └── results/ # Comparações e gráficos
│
├── main_demo.py # Pipeline completo: geração, treino e comparação
├── requirements.txt # Dependências
└── Neural_BVH_Documentacao_Completa.pdf # Documento técnico detalhado