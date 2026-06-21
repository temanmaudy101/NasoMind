# =========================================================
# NasoMind - Aplikasi Skrining Depresi (model biner XGBoost)
# Peran: APLIKASI | Lomba Esai Statistika Nasional
# Nama: ______________________   NIM: ______________
# Program Studi Statistika, Universitas Diponegoro
# =========================================================
# - Antarmuka: Streamlit (aplikasi web Python).
# - Aplikasi TIDAK melatih model. Ia memuat file model (.pkl) + daftar fitur
#   (.json) dari tim Sains Data, lalu memprediksi data yang diunggah.
# - Model bersifat BINER: "Terindikasi depresi" vs "Tidak terindikasi depresi",
#   disertai persentase keyakinan.

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="NasoMind", page_icon="🧠", layout="wide")

# ---------- Gaya tampilan (palet warna branding) ----------
st.markdown(
    """
    <style>
      .stApp { background-color: #FFFFFF; }
      section[data-testid="stSidebar"] { background-color: #A8DF8E; }
      section[data-testid="stSidebar"] * { color: #000000 !important; }

      /* Heading, teks, dan label widget dibuat gelap agar terbaca di latar terang */
      .stApp h2, .stApp h3,
      .stApp [data-testid="stMarkdownContainer"] p,
      .stApp [data-testid="stMarkdownContainer"] li,
      [data-testid="stWidgetLabel"] p,
      [data-testid="stWidgetLabel"] label,
      [data-testid="stMetricLabel"],
      .stTabs [data-baseweb="tab"] p { color: #111111 !important; }

      /* Nilai metric tetap hijau */
      div[data-testid="stMetricValue"] { color: #3E7C2E !important; }

      /* Kotak input: latar putih + teks gelap, apa pun mode browser */
      .stTextInput div[data-baseweb="base-input"],
      .stTextInput div[data-baseweb="input"],
      .stNumberInput div[data-baseweb="input"],
      .stTextArea div[data-baseweb="base-input"] {
          background-color: #FFFFFF !important;
          border: 1px solid #CFE8C2 !important;
      }
      .stTextInput input, .stNumberInput input, .stTextArea textarea {
          color: #111111 !important; -webkit-text-fill-color: #111111 !important;
          background-color: #FFFFFF !important;
      }
      .stTextInput input::placeholder, .stTextArea textarea::placeholder {
          color: #8A8A8A !important; -webkit-text-fill-color: #8A8A8A !important;
      }

      /* Tombol: warna tetap (hijau branding + teks gelap) agar selalu terlihat,
         tidak bergantung pada hover */
      .stButton > button, .stDownloadButton > button,
      [data-testid="stFormSubmitButton"] > button,
      [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-primary"] {
          background-color: #A8DF8E !important;
          color: #111111 !important;
          border: 1px solid #8FCE77 !important;
          font-weight: 600 !important;
      }
      .stButton > button *, [data-testid="stBaseButton-secondary"] *,
      [data-testid="stBaseButton-primary"] * { color: #111111 !important; }
      .stButton > button:hover, .stDownloadButton > button:hover,
      [data-testid="stFormSubmitButton"] > button:hover {
          background-color: #8FCE77 !important;
          border-color: #6FB85A !important;
          color: #111111 !important;
      [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color: #555555 !important; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Konstanta ----------
FILE_MODEL = "xgb_depression_model_Final.pkl"
FILE_FITUR = "stable_features_Final.json"
FILE_HASIL = "hasil_pasien.json"      # menyimpan hasil agar pasien bisa menariknya
LABEL_POS = "Terindikasi depresi"
LABEL_NEG = "Tidak terindikasi depresi"

# Warna hasil (nuansa pastel sesuai palet): positif merah lembut, negatif hijau
WARNA_HASIL = {
    LABEL_POS: {"bg": "#FFBFBF", "fg": "#9A2A2A"},
    LABEL_NEG: {"bg": "#A8DF8E", "fg": "#2F5D1E"},
}

TEKS_KRISIS = (
    "**Layanan dukungan psikologis (gratis, 24 jam):**\n\n"
    "- Hotline **119 ekstensi 8** (SEJIWA / Healing119, Kemenkes)\n"
    "- Situs konselor daring: **www.healing119.id**\n"
    "- Halo Kemenkes: **1500-567**\n\n"
    "Bila kondisi gawat darurat, hubungi **119** atau ke IGD/Puskesmas terdekat."
)
DISCLAIMER = ("⚠️ Hasil berikut merupakan **prediksi**, bukan diagnosis. "
              "Wajib dikonfirmasi kembali oleh psikolog, psikiater, ataupun dokter.")

# ---------- Pengaturan platform Chat & interpretasi GenAI ----------
MODEL_GEMINI = "gemini-2.5-flash"
MODEL_CLAUDE = "claude-sonnet-4-6"
MODEL_OPENAI = "gpt-4o-mini"
SISTEM_PROMPT = (
    "Anda adalah pendamping edukasi kesehatan mental pada aplikasi NasoMind. "
    "Anda BUKAN pengganti tenaga profesional: jangan mendiagnosis, jangan "
    "meresepkan obat. Berikan dukungan empatik dan psikoedukasi singkat seputar "
    "gaya hidup sehat. Untuk hal klinis, sarankan konsultasi ke profesional. Jika "
    "ada tanda krisis/ingin menyakiti diri, arahkan lembut ke 119 ekstensi 8 "
    "(SEJIWA) atau www.healing119.id, tanpa memberi detail menyakiti diri. "
    "Jawab ringkas, hangat, Bahasa Indonesia."
)


# ---------- Memuat model & fitur dari backend ----------
@st.cache_resource
def muat_fitur():
    with open(FILE_FITUR, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def muat_model():
    m = joblib.load(FILE_MODEL)   # membutuhkan paket xgboost terpasang
    # Matikan validasi NAMA fitur pada model. XGBoost menyanitasi karakter "[ ]"
    # saat melatih, sehingga "[8]-Gingerdione" (JSON) berbeda dengan "8-Gingerdione"
    # (model). Urutan fitur tetap sama, jadi kita pakai pencocokan berdasarkan
    # posisi, bukan nama, agar tidak ditolak.
    try:
        m.get_booster().feature_names = None
    except Exception:
        pass
    return m


try:
    FITUR = muat_fitur()
    MODEL = muat_model()
except Exception as e:
    st.error("Gagal memuat model/fitur. Pastikan file '%s' dan '%s' ada di folder "
             "aplikasi dan paket xgboost terpasang.\n\nDetail: %s"
             % (FILE_FITUR, FILE_MODEL, e))
    st.stop()

KELAS_MODEL = list(getattr(MODEL, "classes_", [0, 1]))
# Kelas positif (Terindikasi depresi) diasumsikan berada di indeks ke-1 (mis. label 1).
# Jika ternyata terbalik, ganti IDX_POS menjadi 0.
IDX_POS = 1 if len(KELAS_MODEL) > 1 else 0


# ---------- Penyimpanan hasil (mensimulasikan database backend) ----------
def simpan_hasil(hasil):
    try:
        with open(FILE_HASIL, "w", encoding="utf-8") as f:
            json.dump(hasil, f)
    except Exception:
        pass


def muat_hasil():
    try:
        with open(FILE_HASIL, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# ---------- Skenario solusi (biner) ----------
def solusi_biner(terindikasi):
    if terindikasi:
        rek = ["Disarankan segera berkonsultasi dengan psikolog/psikiater.",
               "Pantau gejala harian dan ceritakan ke orang yang dipercaya.",
               "Dukung dengan gaya hidup sehat (tidur cukup, nutrisi, aktivitas fisik).",
               "Gunakan layanan dukungan di bawah ini bila merasa membutuhkan."]
        return ("Terindikasi - Disarankan Konsultasi", rek, True)
    rek = ["Pertahankan pola tidur 7-9 jam yang teratur.",
           "Jaga pola makan sehat (serat, sayur, makanan fermentasi).",
           "Aktivitas fisik rutin dan jaga koneksi sosial.",
           "Tetap konsultasikan ke profesional bila muncul gejala yang mengganggu."]
    return ("Tidak Terindikasi - Pencegahan & Gaya Hidup", rek, False)


# ---------- Helper GenAI ----------
def ada_kunci(nama):
    try:
        return nama in st.secrets and bool(st.secrets[nama])
    except Exception:
        return False


def jawaban_dari_api(riwayat):
    pesan_terakhir = riwayat[-12:]
    if ada_kunci("GEMINI_API_KEY"):
        from google import genai
        from google.genai import types
        klien = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        isi = []
        for r, t in pesan_terakhir:
            peran_g = "user" if r == "user" else "model"
            isi.append(types.Content(role=peran_g, parts=[types.Part.from_text(text=t)]))
        resp = klien.models.generate_content(
            model=MODEL_GEMINI, contents=isi,
            config=types.GenerateContentConfig(system_instruction=SISTEM_PROMPT,
                                               max_output_tokens=400))
        return resp.text
    if ada_kunci("ANTHROPIC_API_KEY"):
        import anthropic
        klien = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
        pesan = [{"role": r, "content": t} for r, t in pesan_terakhir]
        resp = klien.messages.create(model=MODEL_CLAUDE, max_tokens=400,
                                     system=SISTEM_PROMPT, messages=pesan)
        return resp.content[0].text
    if ada_kunci("OPENAI_API_KEY"):
        from openai import OpenAI
        klien = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        pesan = [{"role": "system", "content": SISTEM_PROMPT}]
        pesan += [{"role": r, "content": t} for r, t in pesan_terakhir]
        resp = klien.chat.completions.create(model=MODEL_OPENAI, max_tokens=400,
                                             messages=pesan)
        return resp.choices[0].message.content
    return None


def jawaban_aturan(p):
    if "tidur" in p:
        return ("Coba jadwal tidur teratur 7-9 jam dan kurangi layar sebelum tidur. "
                "Bila gangguan tidur menetap, konsultasikan ke dokter.")
    if "makan" in p or "diet" in p:
        return ("Perbanyak serat, sayur, dan makanan fermentasi untuk mendukung "
                "kesehatan secara umum. Untuk rencana gizi spesifik, temui ahli gizi.")
    return ("Terima kasih. Untuk hal medis sebaiknya dikonsultasikan dengan "
            "psikolog/psikiater. Ada topik gaya hidup sehat yang ingin ditanyakan?")


def jawaban_ai(pesan, riwayat):
    p = pesan.lower()
    krisis = ["bunuh diri", "mengakhiri", "menyakiti diri", "putus asa",
              "tidak ada harapan", "mengakhiri hidup"]
    if any(k in p for k in krisis):
        return ("Saya khawatir dengan apa yang Anda rasakan, dan Anda tidak "
                "sendirian. Mohon segera hubungi **119 ekstensi 8 (SEJIWA)** atau "
                "buka **www.healing119.id** untuk bicara dengan konselor.")
    try:
        balas = jawaban_dari_api(riwayat)
        if balas:
            return balas
    except Exception:
        return ("(Mode demo) Koneksi ke layanan AI sedang tidak tersedia. "
                "Untuk hal medis, silakan konsultasi ke psikolog/psikiater.")
    return jawaban_aturan(p)


@st.cache_data(show_spinner=False)
def interpretasi_metabolit(daftar):
    """Minta Gemini menjelaskan tiap metabolit (apa & kaitan dengan depresi).
    Dipanggil sekali lalu disimpan di cache. Mengembalikan dict atau None."""
    if not ada_kunci("GEMINI_API_KEY"):
        return None
    try:
        from google import genai
        klien = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        prompt = (
            "Untuk tiap metabolit dalam daftar berikut, jelaskan dalam Bahasa "
            "Indonesia secara SINGKAT dan edukatif: 'apa' (metabolit apa ini) dan "
            "'hubungan' (kemungkinan kaitannya dengan depresi). Jika bukti ilmiahnya "
            "lemah atau tidak pasti, katakan terus terang dan jangan mengarang. "
            "Jangan memberi klaim medis pasti. Balas HANYA JSON dengan struktur "
            '{"nama_metabolit": {"apa": "...", "hubungan": "..."}} tanpa teks lain. '
            "Daftar: " + json.dumps(list(daftar)))
        resp = klien.models.generate_content(model=MODEL_GEMINI, contents=prompt)
        teks = resp.text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(teks)
    except Exception:
        return None


# ---------- Status sesi & login Google ----------
def init_state():
    default = {"peran": None, "df_hasil": None, "id_aktif": None,
               "chat": [], "nama_pasien": None}
    for k, v in default.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


def login_google_aktif():
    try:
        return bool(st.user.is_logged_in)
    except Exception:
        return False


def auth_dikonfigurasi():
    try:
        return "auth" in st.secrets
    except Exception:
        return False


if st.session_state.peran is None and login_google_aktif():
    st.session_state.peran = "pasien"
    try:
        st.session_state.nama_pasien = st.user.name
    except Exception:
        st.session_state.nama_pasien = None


# =========================================================
# HALAMAN LOGIN
# =========================================================
def halaman_login():
    kolom = st.columns([1, 1.4, 1])
    with kolom[1]:
        st.markdown("<h1 style='color:#3E7C2E;text-align:center'>NasoMind</h1>",
                    unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;color:#6B7280'>Deteksi dini "
                    "depresi berbasis mikrobiota & metabolit nasal</p>",
                    unsafe_allow_html=True)
        tab_rs, tab_pasien = st.tabs(["Rumah Sakit", "Pasien"])
        with tab_rs:
            st.text_input("Email", key="rs_email", placeholder="admin@rumahsakit.id")
            st.text_input("Kata sandi", type="password", key="rs_pass",
                          placeholder="Masukkan kata sandi")
            st.write("")
            if st.button("Masuk sebagai Rumah Sakit", use_container_width=True):
                st.session_state.peran = "rumah_sakit"
                st.rerun()
        with tab_pasien:
            st.write("Masuk cukup dengan akun Google Anda, tanpa perlu mengingat "
                     "email/kata sandi.")
            if auth_dikonfigurasi():
                st.button("Lanjutkan dengan Google", use_container_width=True,
                          on_click=st.login)
            else:
                st.caption("Login Google aktif setelah dikonfigurasi di Secrets. "
                           "Untuk uji coba, gunakan tombol di bawah.")
                if st.button("Masuk (mode demo)", use_container_width=True):
                    st.session_state.peran = "pasien"
                    st.rerun()


# =========================================================
# HALAMAN
# =========================================================
def hal_tentang():
    st.header("Tentang Aplikasi")
    with st.container(border=True):
        st.write("**NasoMind** merupakan aplikasi deteksi dini depresi secara presisi berbasis data "
                 "mikrobioma dan metabolit menggunakan model *machine learning* "
                 "untuk transformasi kesehatan Indonesia.")
    with st.container(border=True):
        st.markdown("**Disusun oleh**")
        st.markdown("SD2026020000365\n"
                    "Statistics Essay Competition Satria Data 2026"
        st.caption("Versi prototipe untuk lomba esai statistika nasional.")


def hal_unggah():
    st.header("Unggah Data")
    with st.container(border=True):
        st.write("Unggah file tipe CSV berisi data pasien dengan kolom berikut.")
        with st.expander("Lihat %d nama kolom yang dapat diproses oleh model machine learning" % len(FITUR)):
            st.write(", ".join(FITUR))
        berkas = st.file_uploader("Unggah data")
        if berkas is not None:
            try:
                df = pd.read_csv(berkas)
                kurang = [c for c in FITUR if c not in df.columns]
                if kurang:
                    st.error("Nama kolom belum ada di file: " + ", ".join(kurang))
                    return
                berlebih = [c for c in df.columns
                            if c not in FITUR and c != "ID_Antrian"]
                # SELEKSI variabel sesuai JSON (kolom berlebih otomatis terabaikan).
                # Dikirim sebagai array posisional (.to_numpy) agar XGBoost tidak
                # menolak karena beda penulisan 1 nama fitur antara JSON dan model
                # (urutan ke-20 fitur tetap sama, jadi prediksi tetap benar).
                X = df[FITUR].to_numpy()
                prob = MODEL.predict_proba(X)[:, IDX_POS]   # probabilitas terindikasi
                pred = MODEL.predict(X)                     # label langsung dari model
                kelas_pos = KELAS_MODEL[IDX_POS]            # nilai kelas "terindikasi"

                hasil, label_list, prob_list = {}, [], []
                for i in range(len(df)):
                    p = float(prob[i])
                    label = LABEL_POS if pred[i] == kelas_pos else LABEL_NEG
                    keyakinan = p if label == LABEL_POS else (1 - p)
                    idq = str(df.iloc[i].get("ID_Antrian", "A-%03d" % (i + 1)))
                    hasil[idq] = {"hasil": label, "p": p, "keyakinan": keyakinan}
                    label_list.append(label)
                    prob_list.append(round(p * 100, 1))

                df2 = df.copy()
                df2["Hasil"] = label_list
                df2["Probabilitas (%)"] = prob_list
                st.session_state.df_hasil = df2
                simpan_hasil(hasil)

                pesan = "Data berhasil diunggah. Sebanyak %d kolom data akan diproses oleh model machine learning, sedangkan" % len(df)
                if berlebih:
                    pesan += " %d kolom lainnya akan diabaikan." % len(berlebih)
                st.success(pesan + " Klik menu Dashboard untuk melihat hasil prediksi.")
            except Exception as e:
                st.error("Data gagal diunggah: " + str(e))


def bar_metabolit():
    """Bar chart kontribusi (feature importance) metabolit, dalam persen."""
    try:
        imp = list(MODEL.feature_importances_)
    except Exception:
        return None
    if len(imp) != len(FITUR):
        return None
    total = sum(imp) or 1.0
    pasangan = sorted(zip(FITUR, imp), key=lambda x: x[1], reverse=True)[:8]
    d = pd.DataFrame({"Metabolit": [n for n, _ in pasangan],
                      "Kontribusi (%)": [round(v / total * 100, 1) for _, v in pasangan]})
    fig = px.bar(d, x="Kontribusi (%)", y="Metabolit", orientation="h")
    fig.update_traces(marker_color="#A8DF8E")
    fig.update_layout(height=340, yaxis={"categoryorder": "total ascending"},
                      margin=dict(l=10, r=10, t=10, b=10))
    return fig


def kartu_metabolit():
    """Daftar metabolit teratas + interpretasi AI (apa & kaitan dengan depresi)."""
    try:
        imp = list(MODEL.feature_importances_)
    except Exception:
        return
    if len(imp) != len(FITUR):
        return
    total = sum(imp) or 1.0
    pasangan = sorted(zip(FITUR, imp), key=lambda x: x[1], reverse=True)[:8]
    interp = interpretasi_metabolit(tuple(n for n, _ in pasangan))
    with st.container(border=True):
        st.subheader("Interpretasi Metabolit (AI)")
        if interp is None:
            st.caption("Aktifkan GEMINI_API_KEY di Secrets untuk interpretasi AI.")
        for nama, nilai in pasangan:
            with st.expander("%s — kontribusi %.1f%%" % (nama, nilai / total * 100)):
                if interp and nama in interp:
                    info = interp[nama]
                    st.write("**Apa ini:** " + str(info.get("apa", "-")))
                    st.write("**Kaitan dengan depresi:** " + str(info.get("hubungan", "-")))
                else:
                    st.write("Interpretasi belum tersedia.")
        st.caption("Kontribusi = feature importance model (pengaruh fitur pada model, "
                   "bukan kadar pada pasien). Interpretasi AI bersifat edukatif.")


def fitur_insightful(n=6):
    """Mengambil n fitur paling berpengaruh (insightful) menurut model."""
    try:
        imp = list(MODEL.feature_importances_)
        if len(imp) == len(FITUR):
            return [f for f, _ in sorted(zip(FITUR, imp),
                                         key=lambda x: x[1], reverse=True)[:n]]
    except Exception:
        pass
    return FITUR[:n]


def kartu_statistik(df):
    """Statistik deskriptif dari beberapa fitur paling insightful + box plot."""
    fitur = [f for f in fitur_insightful(6) if f in df.columns]
    if not fitur:
        return
    with st.container(border=True):
        st.subheader("Statistik Deskriptif")
        desk = df[fitur].describe().T[["mean", "std", "min", "50%", "max"]]
        desk.columns = ["Rata-rata", "Std", "Min", "Median", "Maks"]
        st.dataframe(desk.round(3), use_container_width=True)
        dlong = df[fitur].melt(var_name="Fitur", value_name="Nilai")
        fig = px.box(dlong, x="Fitur", y="Nilai", points="outliers")
        fig.update_traces(marker_color="#A8DF8E")
        fig.update_layout(height=340, xaxis_title="",
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Statistik dihitung berdasarkan data yang diunggah. Variabel dipilih dari "
                   "metabolit paling berpengaruh pada model machine learning.")


def kotak_kpi(label, nilai, ikon=""):
    with st.container(border=True):
        st.metric(ikon + " " + label if ikon else label, nilai)


def dashboard_rs(df):
    prob = df["Probabilitas (%)"]
    total = len(df)
    n_pos = int((df["Hasil"] == LABEL_POS).sum())
    pct_pos = n_pos / total * 100 if total else 0
    rata_prob = float(prob.mean()) if total else 0

    # --- Baris KPI (boxes) ---
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kotak_kpi("Jumlah pasien", "%d" % total, "🧑‍🤝‍🧑")
    with k2:
        kotak_kpi("Terindikasi depresi", "%.0f%%" % pct_pos, "🔴")
    with k3:
        kotak_kpi("Tidak terindikasi depresi", "%.0f%%" % (100 - pct_pos), "🟢")
    with k4:
        kotak_kpi("Rata-rata probabilitas", "%.0f%%" % rata_prob, "📊")

    # --- Visualisasi (2 kolom) ---
    v1, v2 = st.columns(2)
    with v1:
        with st.container(border=True):
            st.subheader("Proporsi Pasien")
            d = df["Hasil"].value_counts().reindex([LABEL_NEG, LABEL_POS]).fillna(0)
            fig = px.pie(values=d.values, names=d.index, hole=0.55,
                         color=d.index,
                         color_discrete_map={LABEL_NEG: WARNA_HASIL[LABEL_NEG]["bg"],
                                             LABEL_POS: WARNA_HASIL[LABEL_POS]["bg"]})
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
    with v2:
        with st.container(border=True):
            st.subheader("Sebaran Probabilitas Depresi")
            fig = px.histogram(df, x="Probabilitas (%)", nbins=10)
            fig.update_traces(marker_color="#FFBFBF",
                             marker_line_color="#E26B6B", marker_line_width=1.5)
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                              yaxis_title="Jumlah pasien", bargap=0.1)
            fig.update_xaxes(range=[0, 100], dtick=10)   # mendatar: 0,10,20,...,100
            st.plotly_chart(fig, use_container_width=True)

    # --- Statistik deskriptif fitur ---
    kartu_statistik(df)

    # --- Bar metabolit ---
    fig_m = bar_metabolit()
    if fig_m is not None:
        with st.container(border=True):
            st.subheader("Kontribusi Metabolit pada Model Machine Learning (%)")
            st.plotly_chart(fig_m, use_container_width=True)

    # --- Interpretasi AI ---
    kartu_metabolit()

    # --- Tabel hasil (interaktif) ---
    with st.container(border=True):
        st.subheader("Tabel Hasil Prediksi")
        pilih = st.multiselect("Filter berdasarkan hasil prediksi",
                               [LABEL_NEG, LABEL_POS], default=[LABEL_NEG, LABEL_POS])
        kol = [c for c in ["ID_Antrian", "Hasil Prediksi", "Probabilitas (%)"] if c in df.columns]
        st.dataframe(df[df["Hasil"].isin(pilih)][kol],
                     use_container_width=True, hide_index=True, height=300)


def dashboard_pasien(data):
    label = data.get("hasil", LABEL_NEG)
    p = data.get("p", 0) * 100
    keyakinan = data.get("keyakinan", 0) * 100
    w = WARNA_HASIL.get(label, {"bg": "#EEEEEE", "fg": "#000000"})

    # --- KPI boxes ---
    k1, k2 = st.columns([1.3, 1])
    with k1:
        with st.container(border=True):
            st.caption("Hasil skrining")
            st.markdown("<span style='background:%s;color:%s;padding:6px 18px;"
                        "border-radius:20px;font-weight:700;font-size:18px'>%s</span>"
                        % (w["bg"], w["fg"], label), unsafe_allow_html=True)
            st.write("")
            st.write("Tingkat keyakinan model: **%.0f%%**" % keyakinan)
    with k2:
        with st.container(border=True):
            st.caption("Probabilitas depresi")
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=round(p),
                number={"suffix": "%"},
                gauge={"axis": {"range": [0, 100]},
                       "bar": {"color": w["bg"]}}))
            fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    # --- Rekomendasi ---
    judul, rek, hotline = solusi_biner(label == LABEL_POS)
    with st.container(border=True):
        st.subheader("Rekomendasi: " + judul)
        for r in rek:
            st.markdown("- " + r)
        if hotline:
            st.error(TEKS_KRISIS)

    # --- Metabolit ---
    fig_m = bar_metabolit()
    if fig_m is not None:
        with st.container(border=True):
            st.subheader("Kontribusi Metabolit pada Model (%)")
            st.plotly_chart(fig_m, use_container_width=True)
    kartu_metabolit()


def hal_dashboard():
    st.header("Dashboard")
    st.info(DISCLAIMER)

    if st.session_state.peran == "rumah_sakit":
        if st.session_state.df_hasil is None:
            st.warning("Belum ada data. Silakan unggah data dulu di menu Unggah Data.")
            return
        dashboard_rs(st.session_state.df_hasil)
    else:  # pasien
        with st.container(border=True):
            st.subheader("Cek Hasil Anda")
            idq = st.text_input("Masukkan ID Antrian", placeholder="contoh: A-001")
            if st.button("Lihat Hasil"):
                semua = muat_hasil()
                if idq.strip() in semua:
                    st.session_state.id_aktif = idq.strip()
                else:
                    st.session_state.id_aktif = None
                    st.warning("ID Antrian belum ada. Pastikan Rumah Sakit sudah "
                               "mengunggah data Anda.")
        semua = muat_hasil()
        if st.session_state.id_aktif and st.session_state.id_aktif in semua:
            dashboard_pasien(semua[st.session_state.id_aktif])


def hal_riwayat():
    st.header("Riwayat Konsultasi")
    with st.container(border=True):
        riwayat = pd.DataFrame({
            "Tanggal": ["12 Mei 2026", "02 Apr 2026"],
            "Faskes": ["RS Diponegoro", "RS Diponegoro"],
            "Hasil": ["Terindikasi depresi", "Tidak terindikasi depresi"],
            "Catatan": ["Rujukan psikiater", "Kontrol rutin"],
        })
        st.dataframe(riwayat, use_container_width=True, hide_index=True)


def muat_artikel():
    if not os.path.exists("artikel.md"):
        return []
    teks = open("artikel.md", encoding="utf-8").read()
    daftar, judul, isi = [], None, []
    for baris in teks.splitlines():
        if baris.startswith("## "):
            if judul is not None:
                daftar.append((judul, "\n".join(isi).strip()))
            judul, isi = baris[3:].strip(), []
        elif judul is not None:
            isi.append(baris)
    if judul is not None:
        daftar.append((judul, "\n".join(isi).strip()))
    return daftar


def hal_artikel():
    st.header("Artikel Edukasi")
    daftar = muat_artikel()
    if not daftar:
        st.info("Belum ada artikel. Tambahkan tulisan di file artikel.md.")
        return
    for judul, isi in daftar:
        with st.container(border=True):
            st.subheader(judul)
            st.markdown(isi)


def hal_chat():
    st.header("Chat AI (Pendamping Edukasi)")
    st.caption("Pendamping edukasi, bukan pengganti tenaga profesional.")
    if ada_kunci("GEMINI_API_KEY"):
        st.caption("🟢 Terhubung ke Gemini (Google).")
    elif ada_kunci("ANTHROPIC_API_KEY"):
        st.caption("🟢 Terhubung ke Claude (Anthropic).")
    elif ada_kunci("OPENAI_API_KEY"):
        st.caption("🟢 Terhubung ke ChatGPT (OpenAI).")
    else:
        st.caption("🟡 Mode demo (tanpa API). Tambahkan GEMINI_API_KEY di Secrets.")
    for peran, isi in st.session_state.chat:
        with st.chat_message(peran):
            st.markdown(isi)
    pesan = st.chat_input("Tulis pesan...")
    if pesan:
        st.session_state.chat.append(("user", pesan))
        st.session_state.chat.append(("assistant", jawaban_ai(pesan, st.session_state.chat)))
        st.rerun()


def hal_callcenter():
    st.header("Call Center Depresi")
    with st.container(border=True):
        st.subheader("Anda tidak sendirian.")
        st.write("Jika kamu merasa butuh bantuan atau cerita dengan seseorang, "
                 "klik dan hubungi hotline ini.")
        st.markdown(TEKS_KRISIS)


# =========================================================
# ROUTING UTAMA
# =========================================================
if st.session_state.peran is None:
    halaman_login()
    st.stop()

st.sidebar.title("NasoMind")
if st.session_state.peran == "rumah_sakit":
    st.sidebar.caption("Masuk sebagai Rumah Sakit")
    menu = {"Tentang Aplikasi": hal_tentang, "Unggah Data": hal_unggah,
            "Dashboard": hal_dashboard}
else:
    nama = st.session_state.get("nama_pasien")
    st.sidebar.caption("Masuk sebagai Pasien" + (" (%s)" % nama if nama else ""))
    menu = {"Tentang Aplikasi": hal_tentang, "Dashboard": hal_dashboard,
            "Riwayat Konsultasi": hal_riwayat, "Artikel Edukasi": hal_artikel,
            "Chat AI": hal_chat, "Call Center": hal_callcenter}

pilihan = st.sidebar.radio("Menu", list(menu.keys()))
st.sidebar.divider()
if st.sidebar.button("Keluar"):
    pakai_google = login_google_aktif()
    for k in ["peran", "df_hasil", "id_aktif", "chat", "nama_pasien"]:
        st.session_state[k] = [] if k == "chat" else None
    if pakai_google:
        st.logout()
    else:
        st.rerun()

menu[pilihan]()
