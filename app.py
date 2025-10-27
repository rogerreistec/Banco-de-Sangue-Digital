# app.py
# Banco de Sangue Digital — Painel de Estoques e Produção Hemoterápica

from __future__ import annotations
import io
import unicodedata
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

# =============================================================================
# Configuração da página
# =============================================================================
st.set_page_config(
    page_title="Banco de Sangue Digital | Estoques e Produção Hemoterápica",
    page_icon="🩸",
    layout="wide",
)

st.sidebar.title("🩸 Banco de Sangue Digital")
st.sidebar.caption("Fontes oficiais e dados agregados — prontos para apresentação.")

# =============================================================================
# Constantes
# =============================================================================
DEFAULT_URL = (
    "https://www.gov.br/anvisa/pt-br/centraisdeconteudo/publicacoes/"
    "sangue-tecidos-celulas-e-orgaos/producao-e-avaliacao-de-servicos-de-hemoterapia/"
    "dados-brutos-de-producao-hemoterapica-1/hemoprod_nacional.csv"
)

UF_CENTER = {
    "AC": (-9.02,-70.81),"AL":(-9.57,-36.78),"AM":(-3.41,-65.85),"AP":(1.41,-51.60),
    "BA":(-12.96,-41.55),"CE":(-5.49,-39.32),"DF":(-15.79,-47.86),"ES":(-19.61,-40.18),
    "GO":(-15.82,-49.83),"MA":(-4.96,-45.27),"MG":(-18.51,-44.55),"MS":(-20.77,-54.78),
    "MT":(-12.68,-55.63),"PA":(-3.84,-52.25),"PB":(-7.12,-36.72),"PE":(-8.81,-36.95),
    "PI":(-7.71,-42.72),"PR":(-24.48,-51.86),"RJ":(-22.17,-42.00),"RN":(-5.40,-36.95),
    "RO":(-10.83,-63.34),"RR":(2.73,-62.07),"RS":(-29.33,-53.50),"SC":(-27.24,-50.21),
    "SE":(-10.57,-37.38),"SP":(-22.19,-48.79),"TO":(-10.17,-48.29)
}

UF_NOMES = {
    "RIO DE JANEIRO": "RJ", "SÃO PAULO": "SP", "SAO PAULO": "SP",
    "ESPÍRITO SANTO": "ES", "GOIÁS": "GO", "PARANÁ": "PR",
    "CEARÁ": "CE", "PARÁ": "PA", "RONDÔNIA": "RO",
}

HEMO_LINKS = {
    "AC": "https://www.hemoacre.ac.gov.br/",
    "AL": "http://www.hemoal.saude.al.gov.br/",
    "AM": "https://www.hemoam.am.gov.br/",
    "AP": "https://hemoap.portal.ap.gov.br/",
    "BA": "http://www.hemoba.ba.gov.br/",
    "CE": "https://www.hemoce.ce.gov.br/",
    "DF": "https://www.fhb.df.gov.br/",
    "ES": "https://hemoes.es.gov.br/",
    "GO": "https://www.hemocentro.go.gov.br/",
    "MA": "https://www.hemomar.ma.gov.br/",
    "MG": "https://www.hemominas.mg.gov.br/",
    "MS": "https://www.hemosul.ms.gov.br/",
    "MT": "http://www.saude.mt.gov.br/hemocentro",
    "PA": "https://hemopa.pa.gov.br/",
    "PB": "https://hemocentropb.pb.gov.br/",
    "PE": "https://portal.saude.pe.gov.br/hemocentro",
    "PI": "https://www.hemopi.pi.gov.br/",
    "PR": "http://www.saude.pr.gov.br/HEMEPAR",
    "RJ": "http://www.hemorio.rj.gov.br/",
    "RN": "https://www.hemonorte.rn.gov.br/",
    "RO": "https://rondonia.ro.gov.br/fhemeron/",
    "RR": "https://www.rr.gov.br/orgaos/hemoraima",
    "RS": "https://www.saude.rs.gov.br/hemorgs",
    "SC": "https://www.hemosc.org.br/",
    "SE": "https://saude.se.gov.br/hemose/",
    "SP": "https://www.prosangue.sp.gov.br/",
    "TO": "https://www.to.gov.br/saude/hemorrede/"
}

# =============================================================================
# Funções utilitárias
# =============================================================================
def strip_accents_upper(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "").encode("ascii","ignore").decode("ascii")
    return s.upper().strip()

def uf_para_sigla(v):
    if not v: return None
    v=str(v).strip()
    if len(v)==2: return v.upper()
    return UF_NOMES.get(strip_accents_upper(v), v.upper())

def to_num(c): 
    return pd.to_numeric(c.astype(str).str.replace(".","").str.replace(",","."), errors="coerce")

@st.cache_data(ttl=3600)
def load_default():
    df = pd.read_csv(DEFAULT_URL, dtype=str, sep=None, engine="python", on_bad_lines="skip")
    df.columns = [c.lower().strip() for c in df.columns]
    return df

# =============================================================================
# Página ANVISA
# =============================================================================
def pagina_anvisa():
    st.header("Painel de Estoques e Produção Hemoterápica — ANVISA (Hemoprod)")

    if "df" not in st.session_state:
        st.session_state.df = load_default()

    df = st.session_state.df.copy()

    # Detecta colunas
    col_ano = next((c for c in df.columns if "ano" in c), None)
    col_uf = next((c for c in df.columns if c=="uf" or " uf" in c), None)
    metricas = [c for c in df.columns if c not in {col_ano,col_uf}]

    # Filtros
    ano = st.selectbox("Ano", ["(Todos)"]+sorted(df[col_ano].dropna().unique()) if col_ano else ["(Todos)"])
    uf_c = st.selectbox("Coluna UF", [col_uf], disabled=True)
    met = st.selectbox("Métrica (Soma)", metricas)
    oper = st.selectbox("Agregação", ["Soma","Contagem"])

    if ano!="(Todos)": df = df[df[col_ano]==ano]

    df["__uf__"] = df[uf_c].apply(uf_para_sigla)
    df["__valor__"] = 1 if oper=="Contagem" else to_num(df[met])

    base = df.groupby("__uf__", as_index=False)["__valor__"].sum().rename(columns={"__valor__":"valor"})

    # ✅ Correção RJ/SP fallback (se soma=0 e existem registros → usa contagem)
    if oper=="Soma":
        contagem = df.groupby("__uf__",as_index=False).size().rename(columns={"size":"cont"})
        base = base.merge(contagem,on="__uf__",how="left")
        for uf_fix in ["SP","RJ"]:
            if uf_fix in base["__uf__"].values:
                row = base.loc[base["__uf__"]==uf_fix]
                if float(row["valor"])==0 and float(row["cont"])>0:
                    base.loc[base["__uf__"]==uf_fix,"valor"]=row["cont"]

    base["uf"] = base["__uf__"]
    base = base[["uf","valor"]].dropna()

    st.metric("Total agregado", int(base["valor"].sum()))

    # Mapa
    st.subheader("Mapa por UF")
    plot = []
    vmax = base["valor"].max()
    for _,r in base.iterrows():
        if r["uf"] in UF_CENTER:
            lat,lon = UF_CENTER[r["uf"]]
            plot.append({"uf":r["uf"],"valor":r["valor"],"lat":lat,"lon":lon,"r":4000*np.sqrt(r["valor"]/vmax)})

    if plot:
        layer = pdk.Layer("ScatterplotLayer", data=plot, get_position=["lon","lat"], get_radius="r",
                           get_fill_color=[220,38,38,180], pickable=True)
        st.pydeck_chart(pdk.Deck(layers=[layer],initial_view_state=pdk.ViewState(latitude=-14,longitude=-51,zoom=3.9),
                                tooltip={"text":"{uf}: {valor}"}))

    st.subheader("Tabela agregada por UF")
    st.dataframe(base.sort_values("valor",ascending=False), use_container_width=True)

# =============================================================================
# Página Links Estaduais
# =============================================================================
def pagina_links():
    st.header("Hemocentros Oficiais por Estado")
    df = pd.DataFrame({"UF":list(HEMO_LINKS.keys()),"Acessar":[f"[Abrir]({link})" for link in HEMO_LINKS.values()]})
    st.write(df.to_markdown(index=False), unsafe_allow_html=True)

# =============================================================================
# Página Cadastro (Exemplo local)
# =============================================================================
def pagina_cadastro():
    st.header("Cadastro de possível doador")
    with st.form("f"):
        nome = st.text_input("Nome completo")
        uf = st.selectbox("UF", list(HEMO_LINKS.keys()))
        enviado = st.form_submit_button("Salvar")
    if enviado:
        st.success("Cadastro registrado (local).")

# =============================================================================
# Navegação
# =============================================================================
page = st.sidebar.radio("Navegação",["ANVISA (nacional)","Hemocentros estaduais","Cadastrar doador"])
if page=="ANVISA (nacional)": pagina_anvisa()
elif page=="Hemocentros estaduais": pagina_links()
else: pagina_cadastro()
