"""Lote CEGO de transcrições: escrito depois de calibrar regras e modelo e NUNCA usado para ajustes.

Rótulos: (transcricao, produto, sentimento, churn, oportunidade).
"""

CASOS_CEGOS = [
    ("Atendente: Bom dia, pois não?\nCliente: Oi, o meu investimento em fundos caiu bastante este mês e ninguém me avisou nada. Estou muito preocupado e bravo com isso.\nAtendente: Entendo, vou explicar a oscilação.\nCliente: Se continuar assim vou tirar meu dinheiro e levar para outra corretora.",
     "Investimentos", "negativo", 1, 0),
    ("Atendente: Central, boa tarde.\nCliente: Boa tarde! Só liguei para elogiar a moça que me atendeu semana passada no empréstimo, foi atenciosa demais. Estou tranquilo com as parcelas.\nAtendente: Que bom, vou repassar.",
     "Empréstimo", "positivo", 0, 0),
    ("Atendente: Olá, como ajudo?\nCliente: Preciso da segunda via do boleto da apólice e saber até quando posso pagar sem multa.\nAtendente: Até o dia dez.\nCliente: Beleza, valeu.",
     "Seguro", "neutro", 0, 0),
    ("Atendente: Bom dia!\nCliente: Bom dia, vi no site que vocês têm cartão com cashback. Tenho interesse, como faço para trocar o meu atual?\nAtendente: Posso fazer a solicitação agora.\nCliente: Sim, por favor, gostei da ideia.",
     "Cartão de Crédito", "positivo", 0, 1),
    ("Atendente: Em que posso ajudar?\nCliente: A tarifa mensal da conta subiu e eu não fui comunicado. Isso é falta de respeito com o cliente. Vou pedir o encerramento da conta e abrir em outro banco.\nAtendente: Posso isentar a tarifa.\nCliente: Vou pensar.",
     "Conta Digital", "negativo", 1, 0),
    ("Atendente: Boa noite.\nCliente: Queria confirmar se meu pix de ontem foi compensado. Só isso.\nAtendente: Sim, foi compensado.\nCliente: Obrigado.",
     "Conta Digital", "neutro", 0, 0),
    ("Atendente: Olá!\nCliente: Oi, estou com dificuldade para entender o simulador de financiamento do imóvel, mas gostaria de contratar o crédito se a taxa for boa. Pode me mandar uma proposta?\nAtendente: Envio agora.",
     "Empréstimo", "neutro", 0, 1),
    ("Atendente: Bom dia.\nCliente: Meu seguro de vida cobriu tudo direitinho, fiquei muito satisfeito. Não penso em trocar de seguradora, sou cliente há dez anos.\nAtendente: Agradecemos a confiança.",
     "Seguro", "positivo", 0, 0),
]
