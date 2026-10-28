import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from ortools.sat.python import cp_model


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


def calcular_horarios_turnos(horario_abertura, horario_fechamento):

    formato = "%H:%M"
    abertura = datetime.strptime(horario_abertura, formato)
    fechamento = datetime.strptime(horario_fechamento, formato)
    jornada = timedelta(hours=9, minutes=24)
    
    t1_inicio = abertura
    t1_fim = t1_inicio + jornada
    
    t3_fim = fechamento
    t3_inicio = t3_fim - jornada
    
    meio_tempo = (t3_inicio - t1_inicio) / 2
    t2_inicio = t1_inicio + meio_tempo
    t2_fim = t2_inicio + jornada
    
    return {
        'T1': f"{t1_inicio.strftime(formato)} às {t1_fim.strftime(formato)}",
        'T2': f"{t2_inicio.strftime(formato)} às {t2_fim.strftime(formato)}",
        'T3': f"{t3_inicio.strftime(formato)} às {t3_fim.strftime(formato)}"
    }


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
            for d in dias_semana:
                horario_dia = horarios_loja[d]
                
                # Se a loja abre, calcula as horas reais daquele dia
                mapa_horas = calcular_horarios_turnos(horario_dia[0], horario_dia[1]) if horario_dia else {}
                
                trabalhou = False
                for t in turnos:
                    if solver.Value(x[(b, d, t)]) == 1:
                        # Substitui T1/T2/T3 pela hora real gerada
                        linha[d] = mapa_horas.get(t, t)
                        trabalhou = True
                if not trabalhou:
                    linha[d] = 'FOLGA' if horario_dia is not None else 'FOLGA (Fechada)'
            escala.append(linha)
        return pd.DataFrame(escala)
    else:
        raise ValueError(f"Inviável matematicamente. Com {qtd_balconistas} balconistas e mínimo de {min_simultaneos} por turno, as regras de folga e cobertura não fecham.")


st.set_page_config(page_title="Gerador de Escala", layout="wide")

st.title("Gerador de Escala Automático")
st.markdown("Simulador e otimizador de escalas para as filiais, com horários dinâmicos.")

# SIDEBAR
st.sidebar.header("Parâmetros da Loja")
id_loja_class = {
                1:'HEH',
                2:'FCA',
                3:'BAB',
                4:'IGF',
                5:'HEJ',
                6:'JHG',
                7:'IGF',
                8:'IGK',
                11:'EFL',
                12:'ADK',
                13:'HFA',
                14:'HDA',
                15:'DDA',
                16:'HFA',
                17:'HDA',
                18:'IGF',
                19:'CBC',
                21:'DDA',
                22:'GDA',
                23:'HDH',
                24:'HDA',
                25:'IGF',
                26:'DDA',
                27:'IGF',
                28:'HDA',
                29:'IGF',
                30:'DEA',
                31:'IGD',
                32:'HFL',
                33:'ADA',
                34:'IGF',
                35:'HDI',
                36:'IGE',
                37:'HFE',
                38:'HFK',
                39:'DDA',
                40:'IGF',
                41:'IGF',
                42:'DDK',
                43:'HEH',
                44:'HDA'
                }

id_loja = st.sidebar.selectbox("Filial", id_loja_class)

opcoes_classificacao = ["HEH", "FCA", "BAB", "IGF", "HEJ", "JHG", "IGK", "EFL", 
                        "ADK", "HFA", "HDA", "DDA", "CBC", "GDA", "HDH", "DEA", 
                        "IGD", "HFL", "ADA", "HDI", "IGE", "HFE", "HFK", "DDK"]



if id_loja_class[id_loja] == None:
    classificacao = st.sidebar.selectbox("Classificação de Horário (Origem ETL)", opcoes_classificacao)
else:
    classificacao = id_loja_class[id_loja]
    st.sidebar.text(f"Classificação: {classificacao}")

qtd_balconistas = st.sidebar.selectbox("Balconistas Disponíveis", range(1, 21), index=4)
min_simultaneos = st.sidebar.selectbox("Mínimo Simultâneo na Loja", range(1, 21), index=0)

dias_semana = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
dia_pico = st.sidebar.selectbox("Dia de Pico (Origem ETL)", dias_semana, index=4)

opcoes_turno = {'T1 (Abertura)': 'T1', 'T2 (Intermediário)': 'T2', 'T3 (Fechamento)': 'T3'}
turno_selecionado = st.sidebar.selectbox("Turno de Pico (Origem ETL)", list(opcoes_turno.keys()), index=1)
turno_pico = opcoes_turno[turno_selecionado]

if st.button("Gerar Escala Otimizada", type="primary"):
    with st.spinner("Processando combinações e calculando horários..."):
        try:
            df_escala = gerar_escala_filial(
                id_loja, classificacao, qtd_balconistas, min_simultaneos, dia_pico, turno_pico
            )
            
            st.success("Escala gerada com sucesso!")
            
            def colorir_tabela(val):
                if isinstance(val, str):
                    if "FOLGA" in val:
                        return 'background-color: #c6e0b4; color: #000'
                    elif "às" in val:
                        return 'background-color: #fff2cc; color: #000'
                return ''

            styled_df = df_escala.style.map(colorir_tabela, subset=dias_semana)
            st.dataframe(styled_df, width='stretch', hide_index=True)
            
        except ValueError as erro:
            st.error(f"**Falha na geração:** {erro}")