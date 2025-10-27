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

# =============================================================================
# Constantes
# =============================================================================
DEFAULT_URL = (
    "https://www.gov.br/anvisa/pt-br/centraisdeconteudo/publicacoes/"
    "sangue-tecidos-celulas-e-orgaos/producao-e-avaliacao-de-servicos-de-hemoterapia/"
    "dados-brutos-de-producao-hemoterapica-1/hemoprod_nacional.csv"
)

# centroides aproximados por UF (para o mapa)
UF_CENTER = {
    "AC": (-9.0238, -70.8120), "AL": (-9.5713, -36.7820), "AM": (-3.4168, -65.8561),
    "AP": (1.4156, -51.6022),  "BA": (-12.9694, -41.5556), "CE": (-5.4984, -39.3206),
    "DF": (-15.7998, -47.8645), "ES": (-19.6113, -40.1853), "GO": (-15.8270, -49.8362),
    "MA": (-4.9609, -45.2744), "MG": (-18.5122, -44.5550), "MS": (-20.7722, -54.7852),
    "MT": (-12.6819, -55.6370), "PA": (-3.8431, -52.2500),  "PB": (-7.1219, -36.7240),
    "PE": (-8.8137, -36.9541), "PI": (-7.7183, -42.7289),  "PR": (-24.4842, -51.8625),
    "RJ": (-22.1700, -42.0000), "RN": (-5.4026, -36.9541), "RO": (-10.83, -63.34),
    "RR": (2.7376, -62.0751),  "RS": (-29.3344, -53.5000), "SC": (-27.2423, -50.2189),
    "SE": (-10.5741, -37.3857), "SP": (-22.19, -48.79),     "TO": (-10.1753, -48.2982)
}

# normalização de nomes -> siglas
UF_NOMES = {
    "ACRE": "AC", "ALAGOAS": "AL", "AMAPA": "AP", "AMAPÁ": "AP", "AMAZONAS": "AM",
    "BAHIA": "BA", "CEARA": "CE", "CEARÁ": "CE", "DISTRITO FEDERAL": "DF",
    "ESPIRITO SANTO": "ES", "ESPÍRITO SANTO": "ES", "GOIAS": "GO", "GOIÁS": "GO",
    "MARANHAO": "MA", "MARANHÃO": "MA", "MATO GROSSO DO SUL": "MS",
    "MATO GROSSO": "MT", "MINAS GERAIS": "MG", "PARA": "PA", "PARÁ": "PA",
    "PARAIBA": "PB", "PARAÍBA": "PB", "PARANA": "PR", "PARANÁ": "PR",
    "PERNAMBUCO": "PE", "PIAUI": "PI", "PIAUÍ": "PI", "RIO DE JANEIRO": "RJ",
    "RIO GRANDE DO NORTE": "RN", "RIO GRANDE DO SUL": "RS", "RONDONIA": "RO",
    "RONDÔNIA": "RO", "RORAIMA": "RR", "SANTA CATARINA": "SC",
    "SAO PAULO": "SP", "SÃO PAULO": "SP", "SERGIPE": "SE", "TOCANTINS": "TO"
}

# --------- MAPEAMENTO DE SITES OFICIAIS POR UF (confiáveis) ----------
LINKS_OFICIAIS = {
    "AC": "https://saude.ac.gov.br/hemocentro-do-acre-hemoacre",
    "AL": "http://www.hemoal.al.gov.br/",
    "AM": "https://www.hemoam.am.gov.br/",
    "AP": "https://hemoap.ap.gov.br/",
    "BA": "http://www.hemoba.ba.gov.br/",
    "CE": "https://www.hemoce.ce.gov.br/",
    "DF": "https://www.fhb.df.gov.br/",
    "ES": "https://hemoes.es.gov.br/",
    "GO": "https://www.saude.go.gov.br/hemocentro",
    "MA": "https://www.hemomar.ma.gov.br/",
    "MG": "https://www.hemominas.mg.gov.br/",
    "MS": "https://www.saude.ms.gov.br/hemosul/",
    "MT": "https://www.saude.mt.gov.br/hemocentro",
    "PA": "https://www.hemopa.pa.gov.br/",
    "PB": "https://paraiba.pb.gov.br/diretas/saude/hemocentro",
    "PE": "http://www.hemope.pe.gov.br/",
    "PI": "http://www.hemopi.pi.gov.br/",
    "PR": "https://www.saude.pr.gov.br/Hemepar",
    "RJ": "https://www.hemorio.rj.gov.br/",
    "RN": "http://www.hemonorte.rn.gov.br/",
    "RO": "https://rondonia.ro.gov.br/fhemeron/",
    "RR": "",  # se vazio, cai no fallback
    "RS": "https://saude.rs.gov.br/hemorgs",
    "SC": "https://www.hemosc.org.br/",
    "SE": "https://saude.se.gov.br/hemose/",
    "SP": "https://www.prosangue.sp.gov.br/",
    "TO": "https://www.to.gov.br/saude/hemorrede/",
}
# --------------------------------------------------------------------


# =============================================================================
# Utilidades
# =============================================================================
def strip_accents_upper(s: str) -> str:
    s = unicodedata.normalize("NFD", (s or "")).encode("ascii", "ignore").decode("ascii")
    return " ".join(s.upper().split())

def uf_para_sigla(valor: str) -> str | None:
    if valor is None or str(valor).strip() == "":
        return None
    v = str(valor).strip()
    if len(v) <= 2:  # já é sigla
        return v.upper()
    v2 = strip_accents_upper(v)
    return UF_NOMES.get(v2)

def format_number(x: float) -> str:
    if pd.isna(x):
        return "-"
    try:
        if float(x).is_integer():
            return f"{int(x):,}".replace(",", ".")
        return f"{float(x):,.2f}".replace(",", ".")
    except Exception:
        return str(x)

@st.cache_data(show_spinner=False)
def read_csv_robusto(origem: str | bytes, uploaded: bool = False) -> pd.DataFrame:
    if uploaded:
        byts = origem if isinstance(origem, (bytes, bytearray)) else origem.read()
        buf = io.BytesIO(byts)
    else:
        buf = origem  # URL string

    try:
        df = pd.read_csv(buf, sep=None, engine="python", on_bad_lines="skip", dtype=str)
        if not df.empty:
            return df
    except Exception:
        pass

    try:
        if uploaded:
            buf.seek(0)
        df = pd.read_csv(buf, sep=";", engine="python", on_bad_lines="skip", dtype=str)
        if not df.empty:
            return df
    except Exception:
        pass

    if uploaded:
        buf.seek(0)
    df = pd.read_csv(buf, sep=",", engine="python", on_bad_lines="skip", dtype=str)
    return df

def normaliza_colunas(df: pd.DataFrame) -> pd.DataFrame:
    ren = {c: " ".join(c.strip().lower().split()) for c in df.columns}
    df = df.rename(columns=ren)
    drop_cols = [c for c in df.columns if c.startswith("unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
    return df

def detecta_colunas(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], List[str]]:
    col_ano = None
    for c in df.columns:
        if "ano" in c and ("ref" in c or "refer" in c or "referência" in c):
            col_ano = c
            break

    col_uf = None
    for c in df.columns:
        if c == "uf" or c.endswith(" uf") or c.startswith("uf "):
            col_uf = c
            break

    possiveis = []
    for c in df.columns:
        if c in {col_ano, col_uf}:
            continue
        amostra = pd.to_numeric(
            df[c].dropna().astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
            errors="coerce",
        )
        if amostra.notna().sum() > 0:
            possiveis.append(c)

    if "id da resposta" in df.columns and "id da resposta" in possiveis:
        possiveis = ["id da resposta"] + [x for x in possiveis if x != "id da resposta"]
    return col_ano, col_uf, possiveis

def to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
        errors="coerce",
    )

@st.cache_data(ttl=3600, show_spinner="Carregando dados da ANVISA...")
def get_default_dataframe() -> pd.DataFrame:
    df = read_csv_robusto(DEFAULT_URL, uploaded=False)
    return normaliza_colunas(df)

# =============================================================================
# Páginas
# =============================================================================
def pagina_anvisa():
    st.header("Painel de Estoques e Produção Hemoterápica — ANVISA (Hemoprod)")

    if "hemoprod_df" not in st.session_state:
        st.session_state["hemoprod_df"] = get_default_dataframe()

    with st.expander("Carregar dados (URL ou Upload)", expanded=False):
        url = st.text_input("Cole a URL do CSV aqui", DEFAULT_URL, key="hemoprod_url")
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            carregar = st.button("Carregar URL agora", type="primary")
        with c2:
            up = st.file_uploader("…ou envie o CSV", type=["csv"])
        with c3:
            limpar = st.button("Limpar base e voltar ao padrão")

    if limpar:
        st.session_state.pop("hemoprod_df", None)
        st.session_state["hemoprod_df"] = get_default_dataframe()
        st.success("Base limpa. Dados padrão recarregados.")
        st.rerun()
        return

    if carregar or (up is not None):
        try:
            if up is not None:
                with st.spinner("Lendo arquivo enviado..."):
                    df = read_csv_robusto(up.getvalue(), uploaded=True)
            else:
                with st.spinner(f"Baixando de {url}..."):
                    df = read_csv_robusto(url, uploaded=False)
            df = normaliza_colunas(df)
            st.session_state["hemoprod_df"] = df
            st.success("Base carregada com sucesso!")
            st.rerun()
            return
        except Exception as e:
            st.error(f"Falha ao carregar: {e}")
            return

    df = st.session_state.get("hemoprod_df")
    if df is None or df.empty:
        st.info("O painel está vazio. Tente carregar a URL ou enviar um CSV.")
        return

    col_ano, col_uf, metricas = detecta_colunas(df)

    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.6, 1])
    with c1:
        anos_opc = ["(Todos)"]
        ano_recente = None
        if col_ano and df[col_ano].notna().any():
            anos_num = pd.to_numeric(df[col_ano], errors="coerce")
            if anos_num.notna().any():
                anos_unicos = sorted(list(set(anos_num.dropna().astype(int).tolist())), reverse=True)
                ano_recente = int(anos_num.dropna().max())
                anos_opc = ["(Todos)", "(Mais recente)"] + anos_unicos
        ano_escolhido = st.selectbox("Ano", anos_opc, index=0, key="anv_ano")

    with c2:
        opcoes_uf = ["<não há>"] + list(df.columns)
        default_uf = (df.columns.tolist().index(col_uf) + 1) if (col_uf in df.columns) else 0
        uf_col = st.selectbox("Coluna UF (se existe)", opcoes_uf, index=default_uf, key="anv_uf_col")
        if uf_col == "<não há>":
            uf_col = None

    with c3:
        if len(metricas) == 0:
            metricas = [c for c in df.columns if c not in {col_ano, uf_col}]
        met_col = st.selectbox("Coluna MÉTRICA (para Soma)", metricas, key="anv_metrica")

    with c4:
        oper = st.selectbox("Agregação", ["Soma", "Contagem"], index=0, key="anv_oper")

    df_ag = df.copy()
    if col_ano and ano_escolhido != "(Todos)":
        if ano_escolhido == "(Mais recente)" and (ano_recente is not None):
            df_ag = df_ag[df_ag[col_ano].astype(str) == str(ano_recente)]
        elif ano_escolhido not in {"(Todos)", "(Mais recente)"}:
            df_ag = df_ag[df_ag[col_ano].astype(str) == str(ano_escolhido)]

    total_reg = len(df_ag)
    anos_dist = df_ag[col_ano].nunique(dropna=True) if col_ano else 0
    ufs_dist = df_ag[uf_col].nunique(dropna=True) if uf_col else 0

    if uf_col:
        uf_norm = df_ag[uf_col].astype(str).map(uf_para_sigla)
        uf_norm = uf_norm.fillna(df_ag[uf_col].astype(str).str.upper().str.strip())
        df_ag = df_ag.assign(__uf__=uf_norm)

        if oper == "Soma":
            valores = to_numeric_safe(df_ag[met_col])
            df_ag = df_ag.assign(__valor__=valores)
        else:
            df_ag = df_ag.assign(__valor__=1.0)

        grupo = (
            df_ag.groupby("__uf__", dropna=False, as_index=False)["__valor__"]
            .sum()
            .rename(columns={"__uf__": "uf", "__valor__": "valor"})
        )
        grupo["uf"] = grupo["uf"].astype(str).str.upper().str.strip()
        grupo = grupo[grupo["uf"].isin(UF_CENTER.keys())]

        # Correção apenas para RJ e SP quando Soma zerar
        if oper == "Soma":
            alvo = {"RJ", "SP"}
            zerados = grupo[grupo["uf"].isin(alvo) & ((grupo["valor"].isna()) | (grupo["valor"] == 0))]
            if not zerados.empty:
                contagem = (
                    df_ag[df_ag["__uf__"].isin(alvo)]
                    .groupby("__uf__", dropna=False)
                    .size()
                    .reset_index(name="cont")
                    .rename(columns={"__uf__": "uf"})
                )
                for _, r in zerados.iterrows():
                    uf = r["uf"]
                    cont = contagem.loc[contagem["uf"] == uf, "cont"]
                    if not cont.empty:
                        grupo.loc[grupo["uf"] == uf, "valor"] = float(cont.values[0])
    else:
        grupo = pd.DataFrame(columns=["uf", "valor"])

    total_agregado = float(grupo["valor"].sum()) if len(grupo) else 0.0

    k1, k2, k3, k4b = st.columns(4)
    with k1:
        st.metric("Registros", format_number(total_reg))
    with k2:
        st.metric("Anos distintos", int(anos_dist))
    with k3:
        st.metric("UF distintas", int(ufs_dist))
    with k4b:
        st.metric(("Total (Soma)" if oper == "Soma" else "Total (Contagem)"),
                  format_number(total_agregado))

    st.subheader("Mapa por UF")
    if len(grupo) == 0:
        st.info("Não há dados suficientes para o mapa (verifique a coluna UF e a agregação).")
    else:
        plot_df = []
        vmax = float(grupo["valor"].max() or 1.0)
        for _, r in grupo.iterrows():
            uf = r["uf"]
            val = float(r["valor"]) if pd.notna(r["valor"]) else 0.0
            if uf in UF_CENTER:
                lat, lon = UF_CENTER[uf]
                plot_df.append(dict(uf=uf, valor=val, lat=lat, lon=lon))
        plot_df = pd.DataFrame(plot_df)
        if not plot_df.empty:
            plot_df["radius"] = 6000 + 4000 * np.sqrt(plot_df["valor"] / (vmax if vmax else 1))
            plot_df["valor_formatado"] = plot_df["valor"].apply(format_number)

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=plot_df,
                get_position=["lon", "lat"],
                get_radius="radius",
                get_fill_color=[220, 38, 38, 180],
                pickable=True,
            )
            view_state = pdk.ViewState(latitude=-14.2350, longitude=-51.9253, zoom=3.5)
            tooltip = {"text": "{uf}: {valor_formatado}"}

            st.pydeck_chart(
                pdk.Deck(
                    layers=[layer],
                    initial_view_state=view_state,
                    tooltip=tooltip,
                    map_style="light",
                ),
                use_container_width=True,
            )
            st.caption("🔴 Pontos maiores indicam maior valor agregado (escala raiz).")
        else:
            st.info("Sem pontos válidos para plotar (verifique as UFs).")

    st.subheader("Tabela agregada por UF")
    st.dataframe(grupo.sort_values("valor", ascending=False), use_container_width=True)

    with st.expander(f"Mostrar dados brutos ({len(df)} linhas)", expanded=False):
        st.dataframe(df, use_container_width=True)

def pagina_links_estaduais():
    st.header("Acesse páginas/oficiais e pesquise por hemocentros do seu estado.")
    
    ufs = list(UF_CENTER.keys())
    busca = [f"https://www.google.com/search?q=doar+sangue+{u}+hemocentro" for u in ufs]
    oficiais = [LINKS_OFICIAIS.get(u, "") for u in ufs]

    df_links = pd.DataFrame(
        {
            "UF": ufs,
            "Site oficial": oficiais,
            "Busca (fallback)": busca,
        }
    )

    # Tabela com colunas clicáveis (Streamlit 1.50)
    st.dataframe(
        df_links,
        use_container_width=True,
        column_config={
            "Site oficial": st.column_config.LinkColumn("Site oficial", display_text="Abrir"),
            "Busca (fallback)": st.column_config.LinkColumn("Busca (fallback)", display_text="Abrir"),
        },
    )

    # Botão por UF
    st.subheader("Abrir rapidamente")
    col1, col2 = st.columns([1, 3])
    with col1:
        uf_escolhida = st.selectbox("UF", ufs, index=ufs.index("SP") if "SP" in ufs else 0)
    with col2:
        url_oficial = LINKS_OFICIAIS.get(uf_escolhida) or ""
        url_fallback = f"https://www.google.com/search?q=doar+sangue+{uf_escolhida}+hemocentro"
        url_final = url_oficial if url_oficial else url_fallback
        st.link_button("Abrir site oficial / busca", url_final, type="primary")

def pagina_cadastro():
    st.header("Cadastrar doador (opcional)")
    with st.form("form_doador", clear_on_submit=True):
        c1, c2, c3 = st.columns([2, 2, 1])
        nome = c1.text_input("Nome completo")
        tel = c2.text_input("Telefone/WhatsApp")
        uf = c3.selectbox("UF", sorted(UF_CENTER.keys()))

        c4, c5, c6 = st.columns([2, 2, 1])
        email = c4.text_input("E-mail")
        cidade = c5.text_input("Cidade")
        tipo = c6.selectbox("Tipo sanguíneo", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])

        consent = st.checkbox("Autorizo o uso desses dados para contato sobre doação.")
        enviado = st.form_submit_button("Salvar cadastro", type="primary")

    if enviado:
        if not (nome and email and consent):
            st.warning("Preencha **Nome**, **E-mail** e marque o consentimento.")
        else:
            st.success("Cadastro salvo localmente (exemplo).")
            st.json({"nome": nome, "email": email, "telefone": tel, "uf": uf, "cidade": cidade, "tipo": tipo})

def pagina_rj_sp_experimental():
    st.header("Análise Detalhada: Rio de Janeiro (RJ) e São Paulo (SP) 🩸")
    st.warning(
        "**ATENÇÃO:** Esta seção é **experimental**. Devido à falta de um CSV bruto consolidado "
        "desses estados, o foco aqui é em links diretos para os boletins e a opção de "
        "upload para análise local de dados separados (e não consolidados com a ANVISA)."
    )
    
    st.markdown("---")

    st.subheader("Sites Oficiais e Boletins de Estoque")
    c_rj, c_sp = st.columns(2)
    
    with c_rj:
        st.metric("Rio de Janeiro (RJ)", "HEMORIO")
        st.markdown(
            f"""
            - **Site Principal:** [Abrir Hemorio]({LINKS_OFICIAIS.get('RJ')})
            - **Busca por Boletins/Nível de Estoque:** [Pesquisa Google (Hemorio)](https://www.google.com/search?q=Hemorio+boletim+estoque+de+sangue)
            """
        )
    with c_sp:
        st.metric("São Paulo (SP)", "PROSANGUE")
        st.markdown(
            f"""
            - **Site Principal:** [Abrir Pró-Sangue]({LINKS_OFICIAIS.get('SP')})
            - **Nível de Estoque:** [Página de Níveis da Pró-Sangue](https://www.prosangue.sp.gov.br/estoque/)
            """
        )

    st.markdown("---")
    st.subheader("Upload de Arquivo Bruto (RJ ou SP)")

    up_rj_sp = st.file_uploader(
        "Envie um CSV ou Excel com dados de RJ/SP (para análise local)", 
        type=["csv", "xlsx"]
    )

    if up_rj_sp:
        with st.spinner(f"Lendo arquivo {up_rj_sp.name}..."):
            try:
                # Tenta ler como CSV robusto (se for CSV)
                if up_rj_sp.name.endswith('.csv'):
                    df_upload = read_csv_robusto(up_rj_sp.getvalue(), uploaded=True)
                # Tenta ler como Excel (se for XLSX)
                elif up_rj_sp.name.endswith('.xlsx'):
                    df_upload = pd.read_excel(io.BytesIO(up_rj_sp.getvalue()), dtype=str)
                else:
                    st.error("Formato de arquivo não suportado. Use CSV ou XLSX.")
                    return
                
                df_upload = normaliza_colunas(df_upload)
                
                st.success(f"Arquivo '{up_rj_sp.name}' carregado com sucesso!")
                st.caption(f"Linhas: {len(df_upload)} | Colunas: {len(df_upload.columns)}")
                
                st.subheader("Visualização dos Dados Carregados")
                st.dataframe(df_upload, use_container_width=True)

            except Exception as e:
                st.error(f"Falha ao ler o arquivo. Verifique o formato e a codificação. Erro: {e}")
    else:
        st.info("Aguardando upload de um arquivo CSV/XLSX específico de RJ ou SP.")


def pagina_sobre():
    st.header("Sobre este painel")
    st.markdown(
        """
        **Banco de Sangue Digital** — painel com dados oficiais, pronto para apresentações.

        - **ANVISA (nacional)**: lê o CSV público do Hemoprod, gera KPIs e mapa por UF.  
        - **Hemocentros estaduais**: lista de sites oficiais e busca de fallback.  
        - **RJ/SP (Experimental)**: Links diretos e opção de upload para dados brutos específicos.
        - **Cadastrar doador**: formulário simples (exemplo local).

        **Dica:** use o botão *Carregar URL agora* para atualizar a base diretamente do site da ANVISA
        ou envie o CSV, caso precise trabalhar off-line.
        """
    )

# =============================================================================
# Configuração da Barra Lateral e Navegação
# =============================================================================

# Adiciona um logo simples
st.sidebar.markdown(
    """
    <div style="font-size: 32px; text-align: center; margin-bottom: 5px;">
        <span style="color: red; font-weight: bold;">🩸</span>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("🩸 Banco de Sangue Digital")
st.sidebar.caption("Fontes oficiais e dados agregados — prontos para apresentação.")

st.sidebar.subheader("Navegação")
secao = st.sidebar.radio(
    label="Navegação",
    options=["ANVISA (nacional)", "Hemocentros estaduais", "RJ/SP (Experimental)", "Cadastrar doador", "Sobre"],
    index=0,
    label_visibility="collapsed",
)

st.sidebar.info("💡 Use o botão **Carregar URL agora** para atualizar a base diretamente da ANVISA.")

# Lógica de roteamento
if secao == "ANVISA (nacional)":
    pagina_anvisa()
elif secao == "Hemocentros estaduais":
    pagina_links_estaduais()
elif secao == "RJ/SP (Experimental)":
    pagina_rj_sp_experimental()
elif secao == "Cadastrar doador":
    pagina_cadastro()
else:
    pagina_sobre()
