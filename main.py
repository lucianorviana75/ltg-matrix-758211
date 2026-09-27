import torch
import torch.nn as nn
import torch.optim as optim

print("==========================================")
print("   INICIANDO PROJETO: LTG-MATRIX-758211   ")
print("==========================================\n")

# 1. NOSSOS DADOS (Entradas e Saídas esperadas)
# Exemplo: Entrada [X1, X2] -> Saída [Y = X1 * 2 + X2]
X = torch.tensor([
    [19.75, 1.0],
    [19.82, 2.0],
    [20.11, 3.0],
    [20.26, 4.0]
], dtype=torch.float32)

y = torch.tensor([
    [40.50],
    [41.64],
    [43.22],
    [44.52]
], dtype=torch.float32)

# 2. ARQUITETURA DA REDE NEURAL (LTGMatrixNet)
class LTGMatrixNet(nn.Module):
    def __init__(self):
        super(LTGMatrixNet, self).__init__()
        self.layer1 = nn.Linear(2, 8)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(8, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out

# 3. INICIALIZAÇÃO DO MODELO, CRITÉRIO DE ERRO E OTIMIZADOR
model = LTGMatrixNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("Estrutura do Modelo LTG-Matrix:")
print(model)
print("\nTreinando a IA, aguarde...\n")

# 4. LOOP DE TREINAMENTO (1000 épocas)
for epoch in range(1, 1001):
    predictions = model(X)
    loss = criterion(predictions, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"Época [{epoch}/1000] - Taxa de Erro (Loss): {loss.item():.6f}")

print("\n--- Treinamento Concluído com Sucesso! ---")

# 5. TESTE DE INFERÊNCIA
model.eval()
with torch.no_grad():
    novo_dado = torch.tensor([[20.26, 5.0]], dtype=torch.float32)
    previsao = model(novo_dado)
    print(f"\nEntrada de Teste: [20.26, 5.0]")
    print(f"Previsão gerada pela LTG-Matrix: {previsao.item():.2f}")