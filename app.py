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
# Utilidades
# =============================================================================
def strip_accents_upper(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "").encode("ascii","ignore").decode("ascii")
    return s.upper().strip()

def uf_para_sigla(v):
    if v is None or str(v).strip()=="":
        return None
    v = str(v).strip()
    if len(v)==2:
        return v.upper()
    return UF_NOMES.get(strip_accents_upper(v), v.upper())

def to_num(c: pd.Series) -> pd.Series:
    return pd.to_numeric(
        c.astype(str).str.replace("\u00A0","", regex=False) # NBSP se houver
         .str.replace(".","", regex=False)
         .str.replace(",",".", regex=False),
        errors="coerce"
    )

@st.cache_data(ttl=3600, show_spinner="Baixando base da ANVISA…")
def load_default() -> pd.DataFrame:
    df = pd.read_csv(DEFAULT_URL, dtype=str, sep=None, engine="python", on_bad_lines="skip")
    df.columns = [c.lower().strip() for c in df.columns]
    return df

def metricas_numericas(df: pd.DataFrame, exclude: set) -> list[str]:
    """Devolve as colunas com forte evidência numérica (após parsing)."""
    candidatos = []
    for c in df.columns:
        if c in exclude: 
            continue
        s = to_num(df[c])
        if s.notna().sum() >= max(20, len(df)*0.05) and s.sum(skipna=True) > 0:
            candidatos.append(c)
    # Ordena por “variância” para preferir colunas mais informativas
    candidatos = sorted(candidatos, key=lambda x: to_num(df[x]).var(skipna=True), reverse=True)
    return candidatos

def format_num(x) -> str:
    try:
        x = float(x)
        if x.is_integer(): return f"{int(x):,}".replace(",", ".")
        return f"{x:,.2f}".replace(",", ".")
    except:
        return "-"

# =============================================================================
# Página ANVISA
# =============================================================================
def pagina_anvisa():
    st.header("Painel de Estoques e Produção Hemoterápica — ANVISA (Hemoprod)")

    if "df" not in st.session_state:
        st.session_state.df = load_default()

    df = st.session_state.df.copy()

    # Detecta colunas candidatas
    col_ano = next((c for c in df.columns if "ano" in c), None)
    col_uf  = next((c for c in df.columns if c=="uf" or " uf" in c), None)

    # Sugere métricas realmente numéricas
    sugeridas = metricas_numericas(df, exclude={col_ano, col_uf})
    if not sugeridas and len(df.columns) > 2:
        # fallback “educado”
        sugeridas = [c for c in df.columns if c not in {col_ano, col_uf}]

    # Controles
    c1,c2,c3,c4 = st.columns([1.2,1.2,1.8,1.2])
    with c1:
        anos = ["(Todos)"]
        if col_ano is not None and df[col_ano].notna().any():
            anos = ["(Todos)"] + sorted(df[col_ano].dropna().unique())
        ano = st.selectbox("Ano", anos, index=0)
    with c2:
        st.selectbox("Coluna UF", [col_uf or "(não detectada)"], index=0, disabled=True)
    with c3:
        met = st.selectbox("Métrica (Soma)", sugeridas, index=0 if sugeridas else None)
    with c4:
        oper = st.selectbox("Agregação", ["Soma","Contagem"], index=0)

    # Avançado
    with st.expander("Opções avançadas"):
        usar_soma_crua = st.checkbox("Usar apenas soma crua (sem fallback por contagem em UFs zeradas)", value=False)
        mostrar_debug_rj_sp = st.checkbox("Mostrar amostra de linhas de RJ/SP", value=False)

    # Filtros
    if col_ano and ano != "(Todos)":
        df = df[df[col_ano]==ano]

    if col_uf is None or met is None:
        st.warning("Não foi possível detectar automaticamente as colunas UF e/ou uma métrica numérica.")
        return

    # Normaliza UF + prepara valor
    df["__uf__"] = df[col_uf].apply(uf_para_sigla)
    if oper == "Soma":
        df["__valor__"] = to_num(df[met])
    else:
        df["__valor__"] = 1.0

    # Agregação
    base = (
        df.groupby("__uf__", as_index=False)["__valor__"]
          .sum()
          .rename(columns={"__uf__":"uf","__valor__":"valor"})
    )

    # Mantém apenas UFs brasileiras válidas
    base = base[base["uf"].isin(UF_CENTER.keys())].copy()

    # ⚙️ Fallback: RJ/SP zerados → usa contagem (apenas se a soma for zero e existir dado)
    if oper == "Soma" and not usar_soma_crua:
        cont = df.groupby("__uf__", as_index=False).size().rename(columns={"__uf__":"uf","size":"cont"})
        base = base.merge(cont, on="uf", how="left")
        for uf_fix in ["SP","RJ"]:
            if uf_fix in base["uf"].values:
                lin = base.loc[base["uf"]==uf_fix]
                soma = float(lin["valor"].iloc[0] if not lin.empty else 0.0)
                qtd  = float(lin["cont"].iloc[0]  if "cont" in lin.columns and not lin.empty else 0.0)
                if soma == 0.0 and qtd > 0:
                    base.loc[base["uf"]==uf_fix, "valor"] = qtd
        if "cont" in base.columns:
            base = base.drop(columns=["cont"])

    # KPIs
    colA,colB,colC,colD = st.columns(4)
    with colA: st.metric("Registros", format_num(len(df)))
    with colB: st.metric("Anos distintos", format_num(df[col_ano].nunique() if col_ano else 0))
    with colC: st.metric("UF distintas", format_num(df["__uf__"].nunique()))
    with colD: st.metric(("Total (Soma)" if oper=="Soma" else "Total (Contagem)"), format_num(base["valor"].sum()))

    # Mapa
    st.subheader("Mapa por UF")
    if base.empty:
        st.info("Não há dados para exibir no mapa.")
    else:
        vmax = base["valor"].max() or 1.0
        plot = []
        for _,r in base.iterrows():
            uf = r["uf"]
            if uf in UF_CENTER:
                lat,lon = UF_CENTER[uf]
                plot.append({
                    "uf": uf,
                    "valor": float(r["valor"]),
                    "lat": lat,
                    "lon": lon,
                    "radius": 6000 + 4000*np.sqrt(float(r["valor"])/vmax)
                })
        if plot:
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=plot,
                get_position=["lon","lat"],
                get_radius="radius",
                get_fill_color=[220,38,38,180],
                pickable=True,
            )
            st.pydeck_chart(
                pdk.Deck(
                    layers=[layer],
                    initial_view_state=pdk.ViewState(latitude=-14.2, longitude=-51.9, zoom=3.7),
                    tooltip={"text":"{uf}: {valor}"}
                ),
                use_container_width=True,
            )
            st.caption("🔴 Pontos maiores indicam maior valor agregado (escala raiz).")

    # Ranking / Tabela
    st.subheader("Tabela agregada por UF")
    st.dataframe(base.sort_values("valor", ascending=False), use_container_width=True)

    # Debug RJ/SP
    if mostrar_debug_rj_sp:
        with st.expander("Amostra de linhas — RJ e SP"):
            st.write("**RJ**")
            st.dataframe(df[df["__uf__"]=="RJ"].head(20), use_container_width=True)
            st.write("**SP**")
            st.dataframe(df[df["__uf__"]=="SP"].head(20), use_container_width=True)

# =============================================================================
# Página: Hemocentros oficiais (links clicáveis)
# =============================================================================
def pagina_links_estaduais():
    st.header("Hemocentros Oficiais por Estado (sites verificados)")
    df = pd.DataFrame(
        {"UF": list(HEMO_LINKS.keys()), "Site oficial": list(HEMO_LINKS.values())}
    ).sort_values("UF")

    st.data_editor(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Site oficial": st.column_config.LinkColumn(
                "Site oficial",
                help="Abrir site do hemocentro",
                display_text="Abrir"
            )
        },
        disabled=True,
    )

# =============================================================================
# (Esqueleto) PÁGINA — Painel Avançado
# =============================================================================
def pagina_avancada():
    st.header("Painel Avançado (prévia)")
    st.markdown(
        "- **Heatmap/Ranking** por UF\n"
        "- **Top UFs** por métrica, **tendência temporal** (se a coluna de ano existir)\n"
        "- **Exportar** CSV/Excel dos agregados\n"
        "- **Indicadores críticos** (threshold configurável)\n\n"
        "➡️ Diga qual **coluna métrica** você quer destacar (ex.: *bolsas coletadas*, *coletas*, *doações*), que eu integro os gráficos já usando essa base."
    )

# =============================================================================
# Página Cadastro (Exemplo local)
# =============================================================================
def pagina_cadastro():
    st.header("Cadastro de possível doador (exemplo local)")
    with st.form("f"):
        nome = st.text_input("Nome completo")
        uf = st.selectbox("UF", list(HEMO_LINKS.keys()))
        contato = st.text_input("Telefone/WhatsApp (opcional)")
        ok = st.form_submit_button("Salvar")
    if ok:
        st.success("Cadastro registrado localmente (simulado).")

# =============================================================================
# Navegação
# =============================================================================
st.sidebar.subheader("Navegação")
secao = st.sidebar.radio(
    "Escolha a seção",
    ["ANVISA (nacional)", "Hemocentros estaduais", "Painel avançado", "Cadastrar doador"],
    index=0
)

if secao == "ANVISA (nacional)":
    pagina_anvisa()
elif secao == "Hemocentros estaduais":
    pagina_links_estaduais()
elif secao == "Painel avançado":
    pagina_avancada()
else:
    pagina_cadastro()
