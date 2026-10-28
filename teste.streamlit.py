import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from ortools.sat.python import cp_model

# ==========================================
# 1. TABELAS DE DOMÍNIO
# ==========================================
TABELA_SEG_SEX = {
    'A': ('06:00', '21:00'), 'B': ('06:00', '22:00'), 'C': ('06:00', '23:00'),
    'D': ('06:30', '21:00'), 'E': ('06:30', '22:00'), 'F': ('06:40', '20:00'),
    'G': ('07:00', '20:00'), 'H': ('07:00', '21:00'), 'I': ('07:00', '22:00'),
    'J': ('07:00', '23:00')
}

TABELA_SABADO = {
    'A': ('06:00', '22:00'), 'B': ('06:00', '23:00'), 'C': ('06:40', '19:00'),
    'D': ('07:00', '19:00'), 'E': ('07:00', '20:00'), 'F': ('07:00', '21:00'),
    'G': ('07:00', '22:00'), 'H': ('07:00', '23:00')
}

TABELA_DOMINGO = {
    'A': None,
    'B': ('06:00', '22:00'), 'C': ('06:00', '23:00'), 'D': ('07:00', '19:00'),
    'E': ('07:00', '20:00'), 'F': ('07:00', '22:00'), 'G': ('07:00', '23:00'),
    'H': ('08:00', '14:00'), 'I': ('08:00', '14:10'), 'J': ('08:00', '16:00'),
    'K': ('08:00', '16:20'), 'L': ('08:00', '20:00')
}

def decodificar_classificacao(classificacao):
    c_seg_sex, c_sab, c_dom = classificacao[0], classificacao[1], classificacao[2]
    return {
        'Segunda': TABELA_SEG_SEX[c_seg_sex], 'Terça': TABELA_SEG_SEX[c_seg_sex],
        'Quarta': TABELA_SEG_SEX[c_seg_sex], 'Quinta': TABELA_SEG_SEX[c_seg_sex],
        'Sexta': TABELA_SEG_SEX[c_seg_sex], 'Sábado': TABELA_SABADO[c_sab],
        'Domingo': TABELA_DOMINGO[c_dom]
    }

# ==========================================
# 2. CÁLCULO DINÂMICO DE HORAS (NOVA LÓGICA)
# ==========================================
def calcular_horas_trabalho_fds(horario_loja):
    """Calcula o tempo de janela e trabalho efetivo no FDS, sem estourar o tamanho da loja."""
    formato = "%H:%M"
    abertura = datetime.strptime(horario_loja[0], formato)
    fechamento = datetime.strptime(horario_loja[1], formato)
    duracao_loja = (fechamento - abertura).total_seconds() / 3600.0
    
    if duracao_loja <= 6.25: 
        # Turnos curtos (ex: Domingo 6h). O funcionário trabalha a janela inteira.
        return duracao_loja, duracao_loja 
    elif duracao_loja < 9.4:
        # Loja abre menos que 9h24m, se adapta à loja, reduzindo o efetivo para abater 1h de almoço
        return duracao_loja - 1.0, duracao_loja 
    else:
        # Padrão normal 8h24 de trabalho + 1h de almoço
        return 8.4, 9.4 

def calcular_horarios_turnos(horario_abertura, horario_fechamento, janela_turno_horas):
    """Gera o texto das strings de horas distribuindo a janela dinamicamente"""
    formato = "%H:%M"
    abertura = datetime.strptime(horario_abertura, formato)
    fechamento = datetime.strptime(horario_fechamento, formato)
    
    # Converte horas decimais (ex: 10.2) para horas e minutos exatos (10h12m)
    horas = int(janela_turno_horas)
    minutos = int(round((janela_turno_horas - horas) * 60))
    jornada = timedelta(hours=horas, minutes=minutos)
    
    # Trava de segurança final para nunca passar do horário da loja
    duracao_loja = fechamento - abertura
    if jornada > duracao_loja:
        jornada = duracao_loja
        
    t1_inicio = abertura
    t1_fim = t1_inicio + jornada
    
    t3_fim = fechamento
    t3_inicio = t3_fim - jornada
    
    # Se for uma loja muito curta e t3 iniciar antes de t1, ajusta o meio termo
    meio_tempo = (t3_inicio - t1_inicio) / 2
    t2_inicio = t1_inicio + meio_tempo
    t2_fim = t2_inicio + jornada
    
    return {
        'T1': f"{t1_inicio.strftime(formato)} às {t1_fim.strftime(formato)}",
        'T2': f"{t2_inicio.strftime(formato)} às {t2_fim.strftime(formato)}",
        'T3': f"{t3_inicio.strftime(formato)} às {t3_fim.strftime(formato)}"
    }

# ==========================================
# 3. MOTOR DE OTIMIZAÇÃO (OR-TOOLS)
# ==========================================
def gerar_escala_filial(id_loja, classificacao, qtd_balconistas, min_simultaneos, dia_pico, turno_pico):
    horarios_loja = decodificar_classificacao(classificacao)
    dias_semana = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
    turnos = ['T1', 'T2', 'T3']
    balconistas = [f'Balconista_{i}' for i in range(1, qtd_balconistas + 1)]
    
    model = cp_model.CpModel()
    
    x = {}
    for b in balconistas:
        for d in dias_semana:
            for t in turnos:
                x[(b, d, t)] = model.NewBoolVar(f'x_{b}_{d}_{t}')
                
    objetivo = []
    for b in balconistas:
        objetivo.append(x[(b, dia_pico, turno_pico)] * 10)
        for t in turnos:
            if t != turno_pico:
                objetivo.append(x[(b, dia_pico, t)] * 2)
    model.Maximize(sum(objetivo))
    
    for b in balconistas:
        model.Add(sum(x[(b, d, t)] for d in dias_semana for t in turnos) == 5)
        for d in dias_semana:
            model.Add(sum(x[(b, d, t)] for t in turnos) <= 1)
            
    for d in dias_semana:
        horario = horarios_loja[d]
        if horario is None:
            for b in balconistas:
                for t in turnos:
                    model.Add(x[(b, d, t)] == 0)
        else:
            model.Add(sum(x[(b, d, 'T1')] for b in balconistas) >= min_simultaneos)
            model.Add(sum(x[(b, d, 'T3')] for b in balconistas) >= min_simultaneos)
            
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        escala = []
        for b in balconistas:
            linha = {'Loja': id_loja, 'Classificação': classificacao, 'Balconista': b}
            
            # 1. Mapeia todos os dias que o funcionário trabalha
            dias_trabalhados = []
            for d in dias_semana:
                for t in turnos:
                    if solver.Value(x[(b, d, t)]) == 1:
                        dias_trabalhados.append({'dia': d, 'turno': t, 'horario_loja': horarios_loja[d]})
                        break
                        
            # 2. Paga primeiro as horas do Final de Semana sem estourar a loja
            horas_efetivas_fds = 0.0
            dias_semana_trabalhados = 0
            
            for dt in dias_trabalhados:
                if dt['dia'] in ['Sábado', 'Domingo']:
                    trab_efetivo, janela = calcular_horas_trabalho_fds(dt['horario_loja'])
                    horas_efetivas_fds += trab_efetivo
                    dt['janela_turno'] = janela
                else:
                    dias_semana_trabalhados += 1
                    
            # 3. Distribui as horas restantes nas jornadas de Segunda a Sexta
            horas_restantes = 42.0 - horas_efetivas_fds
            if dias_semana_trabalhados > 0:
                trab_efetivo_semana = horas_restantes / dias_semana_trabalhados
                janela_semana = trab_efetivo_semana + 1.0 # Adiciona a 1h obrigatória de intervalo
            else:
                janela_semana = 9.4
                
            for dt in dias_trabalhados:
                if dt['dia'] not in ['Sábado', 'Domingo']:
                    dt['janela_turno'] = janela_semana
                    
            # 4. Constrói o texto das horas e aloca na linha do DataFrame
            mapa_formatado = {}
            for dt in dias_trabalhados:
                turnos_horarios = calcular_horarios_turnos(dt['horario_loja'][0], dt['horario_loja'][1], dt['janela_turno'])
                mapa_formatado[dt['dia']] = turnos_horarios[dt['turno']]
                
            for d in dias_semana:
                if d in mapa_formatado:
                    linha[d] = mapa_formatado[d]
                else:
                    linha[d] = 'FOLGA' if horarios_loja[d] is not None else 'FOLGA (Fechada)'
                    
            escala.append(linha)
        return pd.DataFrame(escala)
    else:
        raise ValueError(f"Inviável matematicamente. Com {qtd_balconistas} balconistas e mínimo de {min_simultaneos} por turno, as regras de folga e cobertura não fecham.")

# ==========================================
# 4. INTERFACE STREAMLIT
# ==========================================
st.set_page_config(page_title="Gerador de Escala", layout="wide")

st.title("Gerador de Escala Automático")
st.markdown("Simulador e otimizador de escalas para as filiais, com balanço inteligente de horas aos domingos.")

st.sidebar.header("Parâmetros da Loja")

id_loja = st.sidebar.selectbox("ID da Loja", range(1, 45))

opcoes_classificacao = ["HEH", "FCA", "BAB", "IGF", "HEJ", "JHG", "IGK", "EFL", 
                        "ADK", "HFA", "HDA", "DDA", "CBC", "GDA", "HDH", "DEA", 
                        "IGD", "HFL", "ADA", "HDI", "IGE", "HFE", "HFK", "DDK"]
classificacao = st.sidebar.selectbox("Classificação de Horário", opcoes_classificacao)

qtd_balconistas = st.sidebar.selectbox("Balconistas Disponíveis", range(1, 16), index=4)
min_simultaneos = st.sidebar.selectbox("Mínimo Simultâneo na Loja", range(1, 6), index=0)

dias_semana = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
dia_pico = st.sidebar.selectbox("Dia de Pico (Origem ETL)", dias_semana, index=4)

opcoes_turno = {'T1 (Abertura)': 'T1', 'T2 (Intermediário)': 'T2', 'T3 (Fechamento)': 'T3'}
turno_selecionado = st.sidebar.selectbox("Turno de Pico (Origem ETL)", list(opcoes_turno.keys()), index=1)
turno_pico = opcoes_turno[turno_selecionado]

if st.button("Gerar Escala Otimizada", type="primary"):
    with st.spinner("Compensando horas excedentes nos turnos e gerando..."):
        try:
            df_escala = gerar_escala_filial(
                id_loja, classificacao, qtd_balconistas, min_simultaneos, dia_pico, turno_pico
            )
            
            st.success("Escala gerada com sucesso! Turnos aos domingos balanceados.")
            
            def colorir_tabela(val):
                if isinstance(val, str):
                    if "FOLGA" in val:
                        return 'background-color: #c6e0b4; color: #000'
                    elif "às" in val:  
                        return 'background-color: #fff2cc; color: #000'
                return ''

            styled_df = df_escala.style.map(colorir_tabela, subset=dias_semana)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
        except ValueError as erro:
            st.error(f"**Falha na geração:** {erro}")
