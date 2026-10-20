"""Gerador de transcrições sintéticas (pt-BR) de atendimento comercial.

Serve como substituto dos dados reais enquanto eles não estão no repositório.
Para usar dados reais, basta gerar um CSV com as colunas:
    id, transcricao, produto, sentimento, churn, oportunidade
"""
import argparse
import random

import pandas as pd

from . import config

SAUD = [
    "Atendente: Bom dia, em que posso ajudar?",
    "Atendente: Olá, obrigado por entrar em contato. Como posso ajudar hoje?",
    "Atendente: Boa tarde! Aqui é da central de relacionamento, qual o motivo do contato?",
    "Atendente: Oi, tudo bem? Pode me explicar o que aconteceu?",
]

PRODUTO_FRASES = {
    "Conta Digital": [
        "Cliente: Estou ligando sobre a minha conta digital.",
        "Cliente: Tenho uma dúvida sobre o aplicativo da conta corrente, o pix e as transferências.",
        "Cliente: É sobre a conta digital, o saldo e o extrato do app.",
        "Cliente: Preciso falar do pix e da tarifa da minha conta.",
    ],
    "Cartão de Crédito": [
        "Cliente: Estou ligando por causa do meu cartão de crédito.",
        "Cliente: É sobre a fatura do cartão e o limite.",
        "Cliente: Tenho dúvida na anuidade do cartão e nas parcelas da fatura.",
        "Cliente: Meu cartão, a bandeira e o limite de crédito.",
    ],
    "Empréstimo": [
        "Cliente: Estou ligando sobre o meu empréstimo pessoal.",
        "Cliente: É sobre as parcelas do financiamento e a taxa de juros.",
        "Cliente: Quero falar do crédito consignado, o contrato e o saldo devedor.",
        "Cliente: Tenho dúvida na simulação do empréstimo e na quitação.",
    ],
    "Seguro": [
        "Cliente: Estou ligando sobre o meu seguro.",
        "Cliente: É sobre a apólice e a cobertura do seguro de vida.",
        "Cliente: Preciso falar do sinistro e do prêmio do seguro do carro.",
        "Cliente: Tenho dúvida na renovação da apólice e na franquia.",
    ],
    "Investimentos": [
        "Cliente: Estou ligando sobre meus investimentos.",
        "Cliente: É sobre o CDB, a rentabilidade e o resgate.",
        "Cliente: Quero falar da carteira de investimentos, tesouro direto e fundos.",
        "Cliente: Tenho dúvida na corretora e no rendimento da aplicação.",
    ],
}

CORPO = {
    "negativo": [
        "Cliente: Estou muito insatisfeito, já é a terceira vez que ligo e ninguém resolve.",
        "Cliente: Isso é um absurdo, o atendimento é péssimo e demora demais.",
        "Cliente: Fui cobrado indevidamente e ninguém me dá retorno, estou bastante irritado.",
        "Cliente: O sistema vive travando, o aplicativo não funciona e já perdi muito tempo.",
        "Cliente: Estou decepcionado, prometeram resolver e não cumpriram nada.",
        "Cliente: Que falta de respeito, reclamei no protocolo e continua o mesmo problema.",
        "Cliente: As taxas são abusivas e ninguém explica nada, estou revoltado.",
    ],
    "neutro": [
        "Cliente: Queria apenas entender como funciona e confirmar uma informação.",
        "Cliente: Preciso de uma informação sobre prazos e documentos.",
        "Cliente: Gostaria de atualizar meu cadastro e saber os horários de atendimento.",
        "Cliente: Só estou verificando uma data e o valor atual.",
        "Cliente: Pode me enviar o comprovante por e-mail, por favor?",
        "Cliente: Quero saber qual é o procedimento para solicitar a segunda via.",
    ],
    "positivo": [
        "Cliente: Estou muito satisfeito, o atendimento sempre foi excelente.",
        "Cliente: Adoro o serviço, é prático e funciona muito bem, estou contente.",
        "Cliente: Gostei muito da solução que vocês deram, foi rápido e eficiente.",
        "Cliente: Parabéns pelo atendimento, resolveram tudo de forma atenciosa.",
        "Cliente: Recomendo para os amigos, a experiência tem sido ótima.",
        "Cliente: Fiquei feliz com o resultado, muito obrigado pela ajuda.",
    ],
}

# frases "ruído": sentimento contrário/ambíguo para não deixar a tarefa trivial
RUIDO = {
    "negativo": ["Cliente: Apesar de gostar da marca, o problema persiste.",
                 "Cliente: Antes era bom, mas agora está complicado."],
    "neutro": ["Cliente: Não tenho reclamação, mas também não tenho elogio.",
               "Cliente: Está tudo normal, só preciso conferir."],
    "positivo": ["Cliente: Teve uma demora no começo, mas depois resolveram.",
                 "Cliente: Tive um pequeno problema, mas foi superado."],
}

CHURN = [
    "Cliente: Se não resolver hoje vou cancelar tudo.",
    "Cliente: Estou pensando em encerrar a conta e mudar para outro banco.",
    "Cliente: Já fui na concorrência e a proposta deles é melhor, vou migrar.",
    "Cliente: Quero cancelar o serviço e pedir a portabilidade para outra instituição.",
    "Cliente: Vou fechar tudo e procurar outra empresa que respeite o cliente.",
    "Cliente: Cansei, quero o cancelamento imediato e não pretendo continuar.",
]

CHURN_AMBIGUO = [  # fala parecida mas sem intenção real de sair (ruído)
    "Cliente: Não quero cancelar, só quero resolver isso.",
    "Cliente: Já pensei em sair, mas prefiro continuar com vocês.",
]

OPORT = {
    "Conta Digital": ["Cliente: Tenho interesse em conhecer o cartão e um limite maior de pix.",
                      "Cliente: Queria saber se há pacote de serviços com mais benefícios."],
    "Cartão de Crédito": ["Cliente: Gostaria de aumentar o limite e conhecer o cartão black com mais benefícios.",
                          "Cliente: Tenho interesse em um programa de pontos e cashback melhor."],
    "Empréstimo": ["Cliente: Tenho interesse em contratar um novo crédito com taxa menor.",
                   "Cliente: Queria simular um financiamento do imóvel, pode me enviar a proposta?"],
    "Seguro": ["Cliente: Gostaria de contratar também um seguro residencial, pode fazer uma cotação?",
               "Cliente: Tenho interesse em ampliar a cobertura e incluir minha família."],
    "Investimentos": ["Cliente: Quero investir mais, o que vocês têm de produto com melhor rendimento?",
                      "Cliente: Tenho interesse em abrir uma previdência e diversificar a carteira."],
}

FECHO = {
    "negativo": ["Atendente: Entendo, vou escalar o seu caso com prioridade.",
                 "Atendente: Peço desculpas pelo transtorno, vou abrir um protocolo urgente."],
    "neutro": ["Atendente: Certo, a informação foi registrada.",
               "Atendente: Perfeito, enviei os detalhes para o seu e-mail."],
    "positivo": ["Atendente: Fico feliz em ajudar, qualquer coisa estamos à disposição!",
                 "Atendente: Obrigado pelo feedback, vou registrar o elogio."],
}

FILLER = [
    "Atendente: Pode confirmar seu CPF, por favor?",
    "Atendente: Um momento enquanto verifico no sistema.",
    "Cliente: Claro, um instante.",
    "Atendente: Já localizei o seu cadastro.",
]


def gerar_uma(rng: random.Random, i: int) -> dict:
    produto = rng.choice(config.PRODUTOS)
    sent = rng.choices(config.SENTIMENTOS, weights=[0.35, 0.30, 0.35])[0]
    p_churn = {"negativo": 0.50, "neutro": 0.08, "positivo": 0.02}[sent]
    p_oport = {"negativo": 0.05, "neutro": 0.30, "positivo": 0.45}[sent]
    churn = int(rng.random() < p_churn)
    oport = int(rng.random() < p_oport)

    linhas = [rng.choice(SAUD), rng.choice(PRODUTO_FRASES[produto])]
    if rng.random() < 0.12:  # menciona um segundo produto de passagem
        outro = rng.choice([p for p in config.PRODUTOS if p != produto])
        linhas.append(rng.choice(PRODUTO_FRASES[outro]).replace("Estou ligando", "Também tenho"))
    linhas += rng.sample(CORPO[sent], k=rng.randint(1, 2))
    if rng.random() < 0.25:
        linhas.append(rng.choice(RUIDO[sent]))
    if rng.random() < 0.5:
        linhas.append(rng.choice(FILLER))
    if churn:
        linhas.append(rng.choice(CHURN))
    elif rng.random() < 0.06:
        linhas.append(rng.choice(CHURN_AMBIGUO))
    if oport:
        linhas.append(rng.choice(OPORT[produto]))
    linhas.append(rng.choice(FECHO[sent]))

    # ruído de rótulo: 4% de erro de anotação no sentimento, simulando anotação humana
    sent_rot = sent
    if rng.random() < 0.04:
        sent_rot = rng.choice([s for s in config.SENTIMENTOS if s != sent])
    return {"id": i, "transcricao": "\n".join(linhas), "produto": produto,
            "sentimento": sent_rot, "churn": churn, "oportunidade": oport}


def gerar(n: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    return pd.DataFrame([gerar_uma(rng, i) for i in range(n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--treino", type=int, default=800)
    ap.add_argument("--novas", type=int, default=150)
    a = ap.parse_args()
    config.DATA_DIR.mkdir(exist_ok=True)
    gerar(a.treino, config.SEED).to_csv(config.TREINO_CSV, index=False)
    gerar(a.novas, config.SEED + 1000).to_csv(config.NOVAS_SINTETICAS_CSV, index=False)
    print(f"Gerados {a.treino} (treino) e {a.novas} (novas) em {config.DATA_DIR}")


if __name__ == "__main__":
    main()
